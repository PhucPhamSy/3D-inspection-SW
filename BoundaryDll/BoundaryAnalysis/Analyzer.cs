using System;
using System.IO;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using System.Linq;

namespace BoundaryAnalysis
{
    public struct Voxel
    {
        public short Z;
        public short Y;
        public short X;
        public uint Label;
    }

    public static class Analyzer
    {
        [UnmanagedCallersOnly(EntryPoint = "RunBoundaryAnalysis")]
        public static unsafe int RunBoundaryAnalysis(
            byte* bumpMask,
            uint* labeledData,
            int depth, int height, int width,
            float vz, float vy, float vx,
            IntPtr outputDirPtr,
            int* gridRows, int* gridCols)
        {
            try
            {
                string outputDir = Marshal.PtrToStringAnsi(outputDirPtr) ?? string.Empty;
                if (string.IsNullOrEmpty(outputDir)) return -1;

                Directory.CreateDirectory(outputDir);

                // 1. Find Boundary Voxels
                var boundaryLists = new List<Voxel>[depth];
                Parallel.For(0, depth, z =>
                {
                    boundaryLists[z] = new List<Voxel>();
                    if (z == 0 || z == depth - 1) return;

                    int sliceOffset = z * height * width;
                    int prevSlice = (z - 1) * height * width;
                    int nextSlice = (z + 1) * height * width;

                    for (int y = 1; y < height - 1; y++)
                    {
                        int rowOffset = y * width;
                        int prevRow = (y - 1) * width;
                        int nextRow = (y + 1) * width;

                        for (int x = 1; x < width - 1; x++)
                        {
                            int idx = sliceOffset + rowOffset + x;
                            if (bumpMask[idx] > 0 && labeledData[idx] > 0)
                            {
                                // Check 6 face neighbors
                                int hw = height * width;
                                bool isBoundary = (bumpMask[idx - 1] == 0 || bumpMask[idx + 1] == 0 ||
                                                   bumpMask[idx - width] == 0 || bumpMask[idx + width] == 0 ||
                                                   bumpMask[idx - hw] == 0 || bumpMask[idx + hw] == 0);

                                if (isBoundary)
                                {
                                    uint lbl = labeledData[idx];
                                    if (lbl > 0)
                                    {
                                        boundaryLists[z].Add(new Voxel
                                        {
                                            Z = (short)z,
                                            Y = (short)y,
                                            X = (short)x,
                                            Label = lbl
                                        });
                                    }
                                }
                            }
                        }
                    }
                });

                // Merge boundary voxels
                int totalVoxels = boundaryLists.Sum(l => l?.Count ?? 0);
                if (totalVoxels == 0) return 0; // No boundaries

                var allVoxels = new Voxel[totalVoxels];
                int offset = 0;
                for (int z = 0; z < depth; z++)
                {
                    if (boundaryLists[z] == null) continue;
                    boundaryLists[z].CopyTo(allVoxels, offset);
                    offset += boundaryLists[z].Count;
                }

                // 2. Build Spatial Buckets
                float bucketSize = 10.0f;
                int bZ = (int)(depth * vz / bucketSize) + 1;
                int bY = (int)(height * vy / bucketSize) + 1;
                int bX = (int)(width * vx / bucketSize) + 1;

                var buckets = new List<int>[bZ, bY, bX];

                for (int i = 0; i < totalVoxels; i++)
                {
                    var v = allVoxels[i];
                    int zIdx = (int)(v.Z * vz / bucketSize);
                    int yIdx = (int)(v.Y * vy / bucketSize);
                    int xIdx = (int)(v.X * vx / bucketSize);

                    if (buckets[zIdx, yIdx, xIdx] == null)
                        buckets[zIdx, yIdx, xIdx] = new List<int>();
                    buckets[zIdx, yIdx, xIdx].Add(i);
                }

                // 3. Compute Distances
                float[] minDist = new float[totalVoxels];
                float[] distX = new float[totalVoxels];
                float[] distY = new float[totalVoxels];
                float[] distZ = new float[totalVoxels];
                bool[] hasDiff = new bool[totalVoxels];

                Parallel.For(0, totalVoxels, i =>
                {
                    var v = allVoxels[i];
                    float vZ_phys = v.Z * vz;
                    float vY_phys = v.Y * vy;
                    float vX_phys = v.X * vx;

                    int zIdx = (int)(vZ_phys / bucketSize);
                    int yIdx = (int)(vY_phys / bucketSize);
                    int xIdx = (int)(vX_phys / bucketSize);

                    float bestDistSq = float.MaxValue;
                    float bestDx = 0, bestDy = 0, bestDz = 0;
                    bool found = false;

                    int r = 1;
                    int maxR = 4; // Search up to 40um

                    while (!found && r <= maxR)
                    {
                        for (int dz = -r; dz <= r; dz++)
                        {
                            int nZ = zIdx + dz;
                            if (nZ < 0 || nZ >= bZ) continue;
                            for (int dy = -r; dy <= r; dy++)
                            {
                                int nY = yIdx + dy;
                                if (nY < 0 || nY >= bY) continue;
                                for (int dx = -r; dx <= r; dx++)
                                {
                                    int nX = xIdx + dx;
                                    if (nX < 0 || nX >= bX) continue;

                                    var bucket = buckets[nZ, nY, nX];
                                    if (bucket == null) continue;

                                    // AABB Pruning
                                    float cbZ = (nZ + 0.5f) * bucketSize;
                                    float cbY = (nY + 0.5f) * bucketSize;
                                    float cbX = (nX + 0.5f) * bucketSize;

                                    float box_dz = Math.Max(0, Math.Abs(cbZ - vZ_phys) - bucketSize * 0.5f);
                                    float box_dy = Math.Max(0, Math.Abs(cbY - vY_phys) - bucketSize * 0.5f);
                                    float box_dx = Math.Max(0, Math.Abs(cbX - vX_phys) - bucketSize * 0.5f);

                                    if (box_dz * box_dz + box_dy * box_dy + box_dx * box_dx > bestDistSq)
                                        continue;

                                    foreach (int nIdx in bucket)
                                    {
                                        var neighbor = allVoxels[nIdx];
                                        if (neighbor.Label != v.Label)
                                        {
                                            float pdz = Math.Abs(v.Z - neighbor.Z) * vz;
                                            float pdy = Math.Abs(v.Y - neighbor.Y) * vy;
                                            float pdx = Math.Abs(v.X - neighbor.X) * vx;
                                            float dSq = pdz * pdz + pdy * pdy + pdx * pdx;

                                            if (dSq < bestDistSq)
                                            {
                                                bestDistSq = dSq;
                                                bestDz = pdz;
                                                bestDy = pdy;
                                                bestDx = pdx;
                                                found = true;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        r++;
                    }

                    if (found)
                    {
                        hasDiff[i] = true;
                        minDist[i] = (float)Math.Sqrt(bestDistSq);
                        distZ[i] = bestDz;
                        distY[i] = bestDy;
                        distX[i] = bestDx;
                    }
                });

                // 4. Group by label and write CSVs
                var labelGroups = new Dictionary<uint, List<int>>();
                for (int i = 0; i < totalVoxels; i++)
                {
                    if (!labelGroups.TryGetValue(allVoxels[i].Label, out var list))
                    {
                        list = new List<int>();
                        labelGroups[allVoxels[i].Label] = list;
                    }
                    list.Add(i);
                }

                // Super fast CSV writing to prevent GC locks
                using (var writer = new StreamWriter(Path.Combine(outputDir, "boundary_voxel_distances.csv"), false, System.Text.Encoding.ASCII, 65536))
                {
                    writer.WriteLine("Bump_id,voxel_number,Voxel_distance_X,Voxel_distance_Y,Voxel_distance_Z");
                    foreach (var kvp in labelGroups.OrderBy(x => x.Key))
                    {
                        string bId;
                        if (gridRows != null && gridCols != null && gridRows[kvp.Key] != -1 && gridCols[kvp.Key] != -1)
                        {
                            bId = $"\"{gridRows[kvp.Key]},{gridCols[kvp.Key]}\"";
                        }
                        else
                        {
                            bId = "L" + kvp.Key.ToString();
                        }
                        int offsetVox = 0;
                        foreach (int i in kvp.Value)
                        {
                            writer.Write(bId);
                            writer.Write(",");
                            writer.Write(offsetVox.ToString());
                            writer.Write(",");
                            if (hasDiff[i])
                            {
                                writer.Write(distX[i].ToString("F3"));
                                writer.Write(",");
                                writer.Write(distY[i].ToString("F3"));
                                writer.Write(",");
                                writer.Write(distZ[i].ToString("F3"));
                            }
                            else
                            {
                                writer.Write("NaN,NaN,NaN");
                            }
                            writer.WriteLine();
                            offsetVox++;
                        }
                    }
                }

                using (var writer = new StreamWriter(Path.Combine(outputDir, "boundary_summary.csv"), false, System.Text.Encoding.ASCII, 65536))
                {
                    writer.WriteLine("Bump_id,label,boundary_voxels,min_gap_X_um,min_gap_Y_um,min_gap_Z_um,min_gap_euclidean_um,min_gap_voxel_Z,min_gap_voxel_Y,min_gap_voxel_X");
                    foreach (var kvp in labelGroups.OrderBy(x => x.Key))
                    {
                        string bId;
                        if (gridRows != null && gridCols != null && gridRows[kvp.Key] != -1 && gridCols[kvp.Key] != -1)
                        {
                            bId = $"\"{gridRows[kvp.Key]},{gridCols[kvp.Key]}\"";
                        }
                        else
                        {
                            bId = "L" + kvp.Key.ToString();
                        }
                        
                        bool hasAnyDiff = false;
                        float minEucl = float.MaxValue;
                        float mgX = 0, mgY = 0, mgZ = 0;
                        short mz = 0, my = 0, mx = 0;

                        foreach (int i in kvp.Value)
                        {
                            if (hasDiff[i])
                            {
                                hasAnyDiff = true;
                                if (minDist[i] < minEucl)
                                {
                                    minEucl = minDist[i];
                                    mgX = distX[i];
                                    mgY = distY[i];
                                    mgZ = distZ[i];
                                    mz = allVoxels[i].Z;
                                    my = allVoxels[i].Y;
                                    mx = allVoxels[i].X;
                                }
                            }
                        }

                        if (hasAnyDiff)
                        {
                            writer.WriteLine($"{bId},{kvp.Key},{kvp.Value.Count},{mgX:F3},{mgY:F3},{mgZ:F3},{minEucl:F3},{mz},{my},{mx}");
                        }
                        else
                        {
                            writer.WriteLine($"{bId},{kvp.Key},{kvp.Value.Count},NaN,NaN,NaN,NaN,NaN,NaN,NaN");
                        }
                    }
                }

                return 1; // Success
            }
            catch (Exception ex)
            {
                File.WriteAllText(Path.Combine(Marshal.PtrToStringAnsi(outputDirPtr) ?? "C:/", "error.txt"), ex.ToString());
                return -2;
            }
        }
    }
}
