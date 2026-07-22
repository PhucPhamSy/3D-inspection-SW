#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

struct alignas(16) BoundaryVoxelGPU {
    uint16_t z;
    uint16_t y;
    uint16_t x;
    uint16_t pad;
    uint32_t label;
    uint32_t pad2;
};

struct VoxelResultGPU {
    float distX;
    float distY;
    float distZ;
    float minDist;
    bool hasDiff;
};

// 1. Kernel to extract boundary voxels
__global__ void kernel_extract_boundary_voxels(
    const uint8_t *__restrict__ bumpMask,
    const uint32_t *__restrict__ labeledData,
    int depth, int height, int width,
    BoundaryVoxelGPU *out_voxels,
    int *out_count,
    int max_capacity) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= width || y >= height || z >= depth) return;

    size_t idx = (size_t)z * height * width + (size_t)y * width + x;

    if (bumpMask[idx] == 0 || labeledData[idx] == 0) return;

    bool isBoundary = false;
    if (z == 0 || z == depth - 1 || y == 0 || y == height - 1 || x == 0 || x == width - 1) {
        isBoundary = true;
    } else {
        size_t hw = (size_t)height * width;
        if (bumpMask[idx - 1] == 0 || bumpMask[idx + 1] == 0 ||
            bumpMask[idx - width] == 0 || bumpMask[idx + width] == 0 ||
            bumpMask[idx - hw] == 0 || bumpMask[idx + hw] == 0) {
            isBoundary = true;
        }
    }

    if (isBoundary) {
        int pos = atomicAdd(out_count, 1);
        if (pos < max_capacity) {
            BoundaryVoxelGPU v;
            v.z = (uint16_t)z;
            v.y = (uint16_t)y;
            v.x = (uint16_t)x;
            v.pad = 0;
            v.label = labeledData[idx];
            v.pad2 = 0;
            out_voxels[pos] = v;
        }
    }
}

// 2. Kernel to build spatial buckets using atomic linked-list
__global__ void kernel_build_spatial_hash(
    const BoundaryVoxelGPU *voxels,
    int count,
    int *bucketHead,
    int *bucketNext,
    int bZ, int bY, int bX,
    float bucketSize, float vz, float vy, float vx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    BoundaryVoxelGPU v = voxels[i];
    int zIdx = (int)(v.z * vz / bucketSize);
    int yIdx = (int)(v.y * vy / bucketSize);
    int xIdx = (int)(v.x * vx / bucketSize);

    if (zIdx >= 0 && zIdx < bZ && yIdx >= 0 && yIdx < bY && xIdx >= 0 && xIdx < bX) {
        size_t bIdx = (size_t)zIdx * bY * bX + (size_t)yIdx * bX + xIdx;
        int oldHead = atomicExch(&bucketHead[bIdx], i);
        bucketNext[i] = oldHead;
    }
}

// 3. Kernel to compute nearest boundary distances using Spatial Buckets & AABB Pruning
__global__ void kernel_compute_distances_bucket(
    const BoundaryVoxelGPU *voxels,
    int count,
    const int *bucketHead,
    const int *bucketNext,
    int bZ, int bY, int bX,
    float bucketSize, float vz, float vy, float vx,
    VoxelResultGPU *out_results,
    int maxR)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    BoundaryVoxelGPU v = voxels[i];
    float vZ_phys = v.z * vz;
    float vY_phys = v.y * vy;
    float vX_phys = v.x * vx;

    int zIdx = (int)(vZ_phys / bucketSize);
    int yIdx = (int)(vY_phys / bucketSize);
    int xIdx = (int)(vX_phys / bucketSize);

    float bestDistSq = 1e30f;
    float bestDx = 0.0f, bestDy = 0.0f, bestDz = 0.0f;
    bool found = false;

    for (int r = 1; r <= maxR && !found; r++) {
        for (int dz = -r; dz <= r; dz++) {
            int nZ = zIdx + dz;
            if (nZ < 0 || nZ >= bZ) continue;

            for (int dy = -r; dy <= r; dy++) {
                int nY = yIdx + dy;
                if (nY < 0 || nY >= bY) continue;

                for (int dx = -r; dx <= r; dx++) {
                    int nX = xIdx + dx;
                    if (nX < 0 || nX >= bX) continue;

                    // AABB Pruning against bucket bounding box
                    float cbZ = (nZ + 0.5f) * bucketSize;
                    float cbY = (nY + 0.5f) * bucketSize;
                    float cbX = (nX + 0.5f) * bucketSize;

                    float box_dz = fmaxf(0.0f, fabsf(cbZ - vZ_phys) - bucketSize * 0.5f);
                    float box_dy = fmaxf(0.0f, fabsf(cbY - vY_phys) - bucketSize * 0.5f);
                    float box_dx = fmaxf(0.0f, fabsf(cbX - vX_phys) - bucketSize * 0.5f);

                    if (box_dz * box_dz + box_dy * box_dy + box_dx * box_dx > bestDistSq)
                        continue;

                    size_t bIdx = (size_t)nZ * bY * bX + (size_t)nY * bX + nX;
                    int curr = bucketHead[bIdx];

                    while (curr != -1) {
                        BoundaryVoxelGPU neighbor = voxels[curr];
                        if (neighbor.label != v.label) {
                            float pdz = fabsf((float)v.z - (float)neighbor.z) * vz;
                            float pdy = fabsf((float)v.y - (float)neighbor.y) * vy;
                            float pdx = fabsf((float)v.x - (float)neighbor.x) * vx;
                            float dSq = pdz * pdz + pdy * pdy + pdx * pdx;

                            if (dSq < bestDistSq) {
                                bestDistSq = dSq;
                                bestDz = pdz;
                                bestDy = pdy;
                                bestDx = pdx;
                                found = true;
                            }
                        }
                        curr = bucketNext[curr];
                    }
                }
            }
        }
    }

    VoxelResultGPU res;
    if (found) {
        res.distX = bestDx;
        res.distY = bestDy;
        res.distZ = bestDz;
        res.minDist = sqrtf(bestDistSq);
        res.hasDiff = true;
    } else {
        res.distX = 0.0f;
        res.distY = 0.0f;
        res.distZ = 0.0f;
        res.minDist = 0.0f;
        res.hasDiff = false;
    }
    out_results[i] = res;
}

extern "C" {

__declspec(dllexport) int
RunBoundaryAnalysisCUDA(const uint8_t *bumpMask, const uint32_t *labeledData,
                        int depth, int height, int width, float vz, float vy,
                        float vx, const char *outputDir, const int *gridRows,
                        const int *gridCols) 
{
    size_t num_elements = (size_t)depth * height * width;

    uint8_t *d_bumpMask = nullptr;
    uint32_t *d_labeledData = nullptr;
    int *d_boundaryCount = nullptr;
    BoundaryVoxelGPU *d_boundaryVoxels = nullptr;

    cudaError_t err;

    err = cudaMalloc((void **)&d_bumpMask, num_elements * sizeof(uint8_t));
    if (err != cudaSuccess) return -1;

    err = cudaMalloc((void **)&d_labeledData, num_elements * sizeof(uint32_t));
    if (err != cudaSuccess) { cudaFree(d_bumpMask); return -1; }

    cudaMemcpy(d_bumpMask, bumpMask, num_elements * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labeledData, labeledData, num_elements * sizeof(uint32_t), cudaMemcpyHostToDevice);

    int max_capacity = 20000000; // Max 20M boundary voxels (~320MB)
    cudaMalloc((void **)&d_boundaryVoxels, max_capacity * sizeof(BoundaryVoxelGPU));
    cudaMalloc((void **)&d_boundaryCount, sizeof(int));
    cudaMemset(d_boundaryCount, 0, sizeof(int));

    dim3 blockSize3D(32, 8, 1);
    dim3 gridSize3D((width + blockSize3D.x - 1) / blockSize3D.x,
                   (height + blockSize3D.y - 1) / blockSize3D.y,
                   (depth + blockSize3D.z - 1) / blockSize3D.z);

    kernel_extract_boundary_voxels<<<gridSize3D, blockSize3D>>>(
        d_bumpMask, d_labeledData, depth, height, width,
        d_boundaryVoxels, d_boundaryCount, max_capacity);

    cudaDeviceSynchronize();

    int h_boundaryCount = 0;
    cudaMemcpy(&h_boundaryCount, d_boundaryCount, sizeof(int), cudaMemcpyDeviceToHost);

    // Free 3D volume inputs to reclaim VRAM
    cudaFree(d_bumpMask);
    cudaFree(d_labeledData);
    cudaFree(d_boundaryCount);

    if (h_boundaryCount <= 0) {
        cudaFree(d_boundaryVoxels);
        return 0; // No boundaries found
    }

    if (h_boundaryCount > max_capacity) {
        h_boundaryCount = max_capacity;
    }

    // Allocate spatial hashing structures on GPU
    float bucketSize = 10.0f;
    int bZ = (int)(depth * vz / bucketSize) + 1;
    int bY = (int)(height * vy / bucketSize) + 1;
    int bX = (int)(width * vx / bucketSize) + 1;
    size_t totalBuckets = (size_t)bZ * bY * bX;

    int *d_bucketHead = nullptr;
    int *d_bucketNext = nullptr;
    VoxelResultGPU *d_results = nullptr;

    cudaMalloc((void **)&d_bucketHead, totalBuckets * sizeof(int));
    cudaMemset(d_bucketHead, 0xFF, totalBuckets * sizeof(int)); // initialize to -1

    cudaMalloc((void **)&d_bucketNext, h_boundaryCount * sizeof(int));
    cudaMalloc((void **)&d_results, h_boundaryCount * sizeof(VoxelResultGPU));

    int blockSize1D = 256;
    int gridSize1D = (h_boundaryCount + blockSize1D - 1) / blockSize1D;

    kernel_build_spatial_hash<<<gridSize1D, blockSize1D>>>(
        d_boundaryVoxels, h_boundaryCount,
        d_bucketHead, d_bucketNext,
        bZ, bY, bX, bucketSize, vz, vy, vx);

    int maxR = 4; // Search up to 40um (343 buckets instead of 6859)
    kernel_compute_distances_bucket<<<gridSize1D, blockSize1D>>>(
        d_boundaryVoxels, h_boundaryCount,
        d_bucketHead, d_bucketNext,
        bZ, bY, bX, bucketSize, vz, vy, vx,
        d_results, maxR);

    cudaDeviceSynchronize();

    // Allocate Host buffers for results
    std::vector<BoundaryVoxelGPU> h_voxels(h_boundaryCount);
    std::vector<VoxelResultGPU> h_results(h_boundaryCount);

    cudaMemcpy(h_voxels.data(), d_boundaryVoxels, h_boundaryCount * sizeof(BoundaryVoxelGPU), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_results.data(), d_results, h_boundaryCount * sizeof(VoxelResultGPU), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_boundaryVoxels);
    cudaFree(d_bucketHead);
    cudaFree(d_bucketNext);
    cudaFree(d_results);

    // Ultra-fast index sorting by label instead of std::map tree allocations
    std::vector<size_t> indices(h_boundaryCount);
    for (size_t i = 0; i < (size_t)h_boundaryCount; i++) indices[i] = i;
    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return h_voxels[a].label < h_voxels[b].label;
    });

    std::string outDir(outputDir);
    std::string csvPath = outDir + "/boundary_voxel_distances.csv";
    std::string sumPath = outDir + "/boundary_summary.csv";

    char ioBuf[1024 * 1024]; // 1MB I/O buffer

    // Write distances CSV using buffered C stdio
    FILE *fpDist = fopen(csvPath.c_str(), "wb");
    if (!fpDist) return -3;
    setvbuf(fpDist, ioBuf, _IOFBF, sizeof(ioBuf));

    fputs("Bump_id,voxel_number,Voxel_distance_X,Voxel_distance_Y,Voxel_distance_Z\n", fpDist);

    char buf[256];
    size_t i = 0;
    while (i < (size_t)h_boundaryCount) {
        uint32_t lbl = h_voxels[indices[i]].label;
        std::string bId;
        if (gridRows != nullptr && gridCols != nullptr && gridRows[lbl] != -1 && gridCols[lbl] != -1) {
            sprintf(buf, "\"%d,%d\"", gridRows[lbl], gridCols[lbl]);
            bId = buf;
        } else {
            sprintf(buf, "L%u", lbl);
            bId = buf;
        }

        int offsetVox = 0;
        while (i < (size_t)h_boundaryCount && h_voxels[indices[i]].label == lbl) {
            const auto &res = h_results[indices[i]];
            if (res.hasDiff) {
                sprintf(buf, "%s,%d,%.3f,%.3f,%.3f\n", bId.c_str(), offsetVox, res.distX, res.distY, res.distZ);
            } else {
                sprintf(buf, "%s,%d,NaN,NaN,NaN\n", bId.c_str(), offsetVox);
            }
            fputs(buf, fpDist);
            offsetVox++;
            i++;
        }
    }
    fclose(fpDist);

    // Write summary CSV using buffered C stdio
    FILE *fpSum = fopen(sumPath.c_str(), "wb");
    if (!fpSum) return -4;
    setvbuf(fpSum, ioBuf, _IOFBF, sizeof(ioBuf));

    fputs("Bump_id,label,boundary_voxels,min_gap_X_um,min_gap_Y_um,min_gap_Z_um,min_gap_euclidean_um,min_gap_voxel_Z,min_gap_voxel_Y,min_gap_voxel_X\n", fpSum);

    i = 0;
    while (i < (size_t)h_boundaryCount) {
        uint32_t lbl = h_voxels[indices[i]].label;
        std::string bId;
        if (gridRows != nullptr && gridCols != nullptr && gridRows[lbl] != -1 && gridCols[lbl] != -1) {
            sprintf(buf, "\"%d,%d\"", gridRows[lbl], gridCols[lbl]);
            bId = buf;
        } else {
            sprintf(buf, "L%u", lbl);
            bId = buf;
        }

        bool hasAnyDiff = false;
        float minEucl = 1e30f;
        float mgX = 0, mgY = 0, mgZ = 0;
        uint16_t mz = 0, my = 0, mx = 0;
        size_t countVox = 0;

        while (i < (size_t)h_boundaryCount && h_voxels[indices[i]].label == lbl) {
            size_t idx = indices[i];
            const auto &res = h_results[idx];
            const auto &vox = h_voxels[idx];
            if (res.hasDiff) {
                hasAnyDiff = true;
                if (res.minDist < minEucl) {
                    minEucl = res.minDist;
                    mgX = res.distX;
                    mgY = res.distY;
                    mgZ = res.distZ;
                    mz = vox.z;
                    my = vox.y;
                    mx = vox.x;
                }
            }
            countVox++;
            i++;
        }

        if (hasAnyDiff) {
            sprintf(buf, "%s,%u,%zu,%.3f,%.3f,%.3f,%.3f,%d,%d,%d\n", bId.c_str(), lbl,
                    countVox, mgX, mgY, mgZ, minEucl, (int)mz, (int)my, (int)mx);
        } else {
            sprintf(buf, "%s,%u,%zu,NaN,NaN,NaN,NaN,NaN,NaN,NaN\n", bId.c_str(), lbl,
                    countVox);
        }
        fputs(buf, fpSum);
    }
    fclose(fpSum);

    return 1;
}

} // extern "C"
