using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace ReconSimCS;

[StructLayout(LayoutKind.Sequential, Pack = 1)]
public struct RecvPacket
{
    public int Counter;
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 512)]
    public byte[] Path;
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 4)]
    public byte[] Flags;
} // 520 bytes

public partial class Form1 : Form
{
    private const int RECV_PACKET_SIZE = 520;
    private const int SEND_PACKET_SIZE = 528;
    // CPU + MEM + Disk[A..J] = 12 * uint64 (ULONGLONG), little-endian
    private const int FDC_PACKET_SIZE = 96;

    private CancellationTokenSource? _fdcCts;
    private Task? _fdcListenerTask;
    private TcpListener? _fdcListener;
    private int _fdcPacketCount;

    public Form1()
    {
        InitializeComponent();
        SetupGrid();
        this.FormClosing += Form1_FormClosing;
    }

    private void SetupGrid()
    {
        gridResults.Columns.Add("colTime", "Time");
        gridResults.Columns.Add("colCounter", "Counter");
        gridResults.Columns.Add("colSpeed", "File Speed (MB/s)");
        gridResults.Columns.Add("colConn", "Conn (ms)");
        gridResults.Columns.Add("colACK", "ACK (ms)");
        gridResults.Columns.Add("colRTT", "Total RTT (ms)");
    }

    private void Log(string message, Color color)
    {
        if (InvokeRequired)
        {
            Invoke(new Action(() => Log(message, color)));
            return;
        }

        string timeStr = DateTime.Now.ToString("HH:mm:ss");
        txtLog.SelectionStart = txtLog.TextLength;
        txtLog.SelectionLength = 0;
        txtLog.SelectionColor = Color.Gray;
        txtLog.AppendText($"[{timeStr}] ");
        txtLog.SelectionColor = color;
        txtLog.AppendText(message + "\n");
        txtLog.ScrollToCaret();
    }

    private void SetFdcPacketCount(int count)
    {
        if (InvokeRequired)
        {
            Invoke(new Action(() => SetFdcPacketCount(count)));
            return;
        }
        lblFdcPackets.Text = $"Packets: {count}";
    }

    private void btnBrowseFile_Click(object sender, EventArgs e)
    {
        using OpenFileDialog openFileDialog = new OpenFileDialog();
        openFileDialog.InitialDirectory = @"Z:\";
        openFileDialog.Filter = "TIFF Files (*.tif;*.tiff)|*.tif;*.tiff|All Files (*.*)|*.*";
        if (openFileDialog.ShowDialog() == DialogResult.OK)
        {
            txtPath.Text = openFileDialog.FileName;
        }
    }

    private void btnBrowseFolder_Click(object sender, EventArgs e)
    {
        using FolderBrowserDialog folderBrowserDialog = new FolderBrowserDialog();
        folderBrowserDialog.InitialDirectory = @"Z:\";
        if (folderBrowserDialog.ShowDialog() == DialogResult.OK)
        {
            txtPath.Text = folderBrowserDialog.SelectedPath;
        }
    }

    private async void btnFdcListen_Click(object sender, EventArgs e)
    {
        try
        {
            if (_fdcListenerTask is null)
            {
                await StartFdcListenerAsync();
                return;
            }
            await StopFdcListenerAsync("FDC listener stopped by user.");
        }
        catch (Exception ex)
        {
            Log($"FDC listener toggle failed: {ex.Message}", Color.Red);
            lblStatus.Text = "FDC listener error.";
        }
    }

    private async Task StartFdcListenerAsync()
    {
        if (_fdcListenerTask is not null)
        {
            return;
        }

        try
        {
            int port = (int)numFdcPort.Value;
            _fdcPacketCount = 0;
            SetFdcPacketCount(0);

            _fdcCts = new CancellationTokenSource();
            _fdcListener = new TcpListener(IPAddress.Any, port);
            _fdcListener.Start();

            btnFdcListen.Text = "Stop FDC Listener";
            btnFdcListen.BackColor = Color.FromArgb(200, 60, 60);
            lblStatus.Text = $"FDC listener started on 0.0.0.0:{port}";
            Log($"FDC LISTENER STARTED on 0.0.0.0:{port}", Color.DeepSkyBlue);

            CancellationToken token = _fdcCts.Token;
            _fdcListenerTask = Task.Run(() => RunFdcListenerLoopAsync(token), token);
            await Task.CompletedTask;
        }
        catch
        {
            try
            {
                _fdcCts?.Cancel();
                _fdcListener?.Stop();
            }
            catch
            {
                // ignore cleanup errors
            }
            _fdcCts = null;
            _fdcListener = null;
            _fdcListenerTask = null;
            btnFdcListen.Text = "Start FDC Listener";
            btnFdcListen.BackColor = Color.FromArgb(36, 122, 213);
            throw;
        }
    }

    private async Task StopFdcListenerAsync(string reason)
    {
        CancellationTokenSource? cts = _fdcCts;
        Task? listenerTask = _fdcListenerTask;
        TcpListener? listener = _fdcListener;

        _fdcCts = null;
        _fdcListenerTask = null;
        _fdcListener = null;

        try
        {
            cts?.Cancel();
        }
        catch
        {
            // ignore cancellation edge cases
        }
        try
        {
            listener?.Stop();
        }
        catch
        {
            // ignore listener stop edge cases
        }

        if (listenerTask is not null)
        {
            try
            {
                await listenerTask;
            }
            catch (OperationCanceledException)
            {
                // expected on normal stop
            }
            catch (ObjectDisposedException)
            {
                // expected when listener socket is closed
            }
            catch (Exception ex)
            {
                Log($"FDC listener stop warning: {ex.Message}", Color.Orange);
            }
        }

        btnFdcListen.Text = "Start FDC Listener";
        btnFdcListen.BackColor = Color.FromArgb(36, 122, 213);
        lblStatus.Text = reason;
        Log(reason, Color.Orange);
    }

    private async Task RunFdcListenerLoopAsync(CancellationToken token)
    {
        if (_fdcListener is null)
        {
            return;
        }

        while (!token.IsCancellationRequested)
        {
            TcpClient? client = null;
            try
            {
                client = await _fdcListener.AcceptTcpClientAsync(token);
                IPEndPoint? remote = client.Client.RemoteEndPoint as IPEndPoint;
                Log(
                    $"FDC client connected: {remote?.Address}:{remote?.Port}",
                    Color.LightSkyBlue
                );
                await HandleFdcClientAsync(client, token);
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (ObjectDisposedException)
            {
                break;
            }
            catch (Exception ex)
            {
                if (!token.IsCancellationRequested)
                {
                    Log($"FDC listener error: {ex.Message}", Color.OrangeRed);
                }
            }
            finally
            {
                try
                {
                    client?.Close();
                }
                catch
                {
                    // ignore socket close race
                }
            }
        }
    }

    private async Task HandleFdcClientAsync(TcpClient client, CancellationToken token)
    {
        using NetworkStream stream = client.GetStream();
        byte[] packet = new byte[FDC_PACKET_SIZE];

        while (!token.IsCancellationRequested)
        {
            bool gotPacket = await ReadExactlyAsync(stream, packet, token);
            if (!gotPacket)
            {
                Log("FDC client disconnected.", Color.DarkGray);
                return;
            }

            // Scale confirmed by Mr Nam: 99.99% -> 9999 (percent * 100)
            ulong cpuRaw = BitConverter.ToUInt64(packet, 0);
            ulong memRaw = BitConverter.ToUInt64(packet, 8);
            // Disk[0]=A ... Disk[9]=J; display only C..J
            ulong[] disks = new ulong[10];
            for (int i = 0; i < 10; i++)
            {
                disks[i] = BitConverter.ToUInt64(packet, 16 + i * 8);
            }

            int count = Interlocked.Increment(ref _fdcPacketCount);
            SetFdcPacketCount(count);
            // Match Mr Nam OutputDebugString style (raw ULONGLONG), plus human % for check.
            Log(
                $"FDC #{count:0000} "
                + $"CPU={cpuRaw:D4} MEM={memRaw:D4} "
                + $"C={disks[2]:D4} D={disks[3]:D4} E={disks[4]:D4} F={disks[5]:D4} "
                + $"G={disks[6]:D4} H={disks[7]:D4} I={disks[8]:D4} J={disks[9]:D4} "
                + $"| CPU={cpuRaw / 100.0:F2}% MEM={memRaw / 100.0:F2}% "
                + $"C={disks[2] / 100.0:F2}% D={disks[3] / 100.0:F2}% "
                + $"E={disks[4] / 100.0:F2}% F={disks[5] / 100.0:F2}%",
                Color.LightGreen
            );
        }
    }

    private static async Task<bool> ReadExactlyAsync(
        NetworkStream stream,
        byte[] buffer,
        CancellationToken token
    )
    {
        int totalRead = 0;
        while (totalRead < buffer.Length)
        {
            int read = await stream.ReadAsync(
                buffer.AsMemory(totalRead, buffer.Length - totalRead),
                token
            );
            if (read == 0)
            {
                return false;
            }
            totalRead += read;
        }
        return true;
    }

    private async void btnSend_Click(object sender, EventArgs e)
    {
        string ip = txtIP.Text.Trim();
        int port = (int)numPort.Value;
        string volumePath = txtPath.Text.Trim();
        int counter = (int)numCounter.Value;
        bool[] flags = new bool[] { chkFlag0.Checked, chkFlag1.Checked, chkFlag2.Checked, chkFlag3.Checked };

        if (string.IsNullOrEmpty(volumePath))
        {
            Log("Error: Please select a valid volume path!", Color.Red);
            return;
        }

        btnSend.Enabled = false;
        lblStatus.Text = $"Transmitting to {ip}:{port}...";
        Log(new string('=', 60), Color.DarkGray);
        Log($"Starting test for Counter #{counter} -> Path: '{volumePath}'", Color.Cyan);

        await Task.Run(() => PerformBenchmarkAndSend(ip, port, volumePath, counter, flags));

        btnSend.Enabled = true;
    }

    private void PerformBenchmarkAndSend(string ip, int port, string volumePath, int counter, bool[] flags)
    {
        double fileCheckMs = 0;
        double readSpeedMbps = 0;
        double connMs = 0;
        double sendMs = 0;
        double ackMs = 0;
        double totalRttMs = 0;
        bool success = false;

        // 1. Measure File Access Speed
        Log($"Checking volume file access: '{volumePath}'...", Color.Cyan);
        Stopwatch sw = Stopwatch.StartNew();
        bool exists = File.Exists(volumePath) || Directory.Exists(volumePath);
        sw.Stop();
        fileCheckMs = sw.Elapsed.TotalMilliseconds;

        if (exists && File.Exists(volumePath))
        {
            try
            {
                FileInfo fi = new FileInfo(volumePath);
                double sizeMb = fi.Length / (1024.0 * 1024.0);
                Log($"File exists ({sizeMb:F2} MB). Measuring throughput...", Color.LightGreen);

                sw.Restart();
                long totalRead = 0;
                byte[] buffer = new byte[10 * 1024 * 1024]; // 10MB chunk
                using (FileStream fs = File.OpenRead(volumePath))
                {
                    int bytesRead;
                    while ((bytesRead = fs.Read(buffer, 0, buffer.Length)) > 0)
                    {
                        totalRead += bytesRead;
                        if (totalRead >= 200 * 1024 * 1024) break; // cap at 200MB
                    }
                }
                sw.Stop();
                double readSec = sw.Elapsed.TotalSeconds;
                double readMb = totalRead / (1024.0 * 1024.0);
                readSpeedMbps = readSec > 0 ? readMb / readSec : 0;
                Log($"Throughput: {readSpeedMbps:F2} MB/s (Read {readMb:F1} MB in {readSec:F3} s)", Color.GreenYellow);
            }
            catch (Exception ex)
            {
                Log($"Warning reading file: {ex.Message}", Color.Orange);
            }
        }
        else
        {
            Log("Warning: File/Folder path does not exist locally or drive is unmapped!", Color.Orange);
        }

        // 2. Perform TCP Communication
        Log($"Connecting to Inspection PC at {ip}:{port}...", Color.DodgerBlue);
        try
        {
            Stopwatch swTotal = Stopwatch.StartNew();
            using TcpClient client = new TcpClient();
            
            sw.Restart();
            client.Connect(ip, port);
            sw.Stop();
            connMs = sw.Elapsed.TotalMilliseconds;
            Log($"TCP Connected in {connMs:F3} ms", Color.LightGreen);

            using NetworkStream stream = client.GetStream();

            // Build 520-byte packet
            byte[] packet = new byte[RECV_PACKET_SIZE];
            byte[] counterBytes = BitConverter.GetBytes(counter);
            Array.Copy(counterBytes, 0, packet, 0, 4);

            byte[] pathBytes = Encoding.UTF8.GetBytes(volumePath);
            int pathLen = Math.Min(pathBytes.Length, 511);
            Array.Copy(pathBytes, 0, packet, 4, pathLen);

            for (int i = 0; i < 4; i++)
            {
                packet[516 + i] = flags[i] ? (byte)1 : (byte)0;
            }

            sw.Restart();
            stream.Write(packet, 0, packet.Length);
            sw.Stop();
            sendMs = sw.Elapsed.TotalMilliseconds;
            Log($"Sent 520-byte TCP trigger in {sendMs:F3} ms", Color.LightGreen);

            // Receive 528-byte ACK
            sw.Restart();
            byte[] ackBuffer = new byte[SEND_PACKET_SIZE];
            int totalRecv = 0;
            while (totalRecv < SEND_PACKET_SIZE)
            {
                int read = stream.Read(ackBuffer, totalRecv, SEND_PACKET_SIZE - totalRecv);
                if (read == 0) break;
                totalRecv += read;
            }
            sw.Stop();
            ackMs = sw.Elapsed.TotalMilliseconds;
            swTotal.Stop();
            totalRttMs = swTotal.Elapsed.TotalMilliseconds;

            if (totalRecv == SEND_PACKET_SIZE)
            {
                int resCounter = BitConverter.ToInt32(ackBuffer, 0);
                int status0 = BitConverter.ToInt32(ackBuffer, 516);
                int status1 = BitConverter.ToInt32(ackBuffer, 520);
                int status2 = BitConverter.ToInt32(ackBuffer, 524);
                success = true;

                Log($"SUCCESS: Received ACK in {ackMs:F3} ms (Total RTT: {totalRttMs:F3} ms, Status: [{status0},{status1},{status2}])", Color.SpringGreen);
            }
            else
            {
                Log($"ERROR: Incomplete ACK packet ({totalRecv}/528 bytes)", Color.Red);
            }
        }
        catch (Exception ex)
        {
            Log($"TCP ERROR: {ex.Message}", Color.Red);
        }

        UpdateUIResults(counter, readSpeedMbps, connMs, ackMs, totalRttMs, success);
    }

    private void UpdateUIResults(int counter, double speed, double connMs, double ackMs, double rttMs, bool success)
    {
        if (InvokeRequired)
        {
            Invoke(new Action(() => UpdateUIResults(counter, speed, connMs, ackMs, rttMs, success)));
            return;
        }

        string timeStr = DateTime.Now.ToString("HH:mm:ss");
        int rowIndex = gridResults.Rows.Add(timeStr, counter, $"{speed:F1} MB/s", $"{connMs:F2} ms", $"{ackMs:F2} ms", $"{rttMs:F2} ms");

        if (success)
        {
            gridResults.Rows[rowIndex].Cells[5].Style.ForeColor = Color.Green;
            numCounter.Value = numCounter.Value + 1;
            lblStatus.Text = $"Transmission OK! Total RTT: {rttMs:F2} ms | Read Speed: {speed:F1} MB/s";
        }
        else
        {
            gridResults.Rows[rowIndex].Cells[5].Style.ForeColor = Color.Red;
            lblStatus.Text = "Transmission Failed!";
        }
    }

    private void Form1_FormClosing(object? sender, FormClosingEventArgs e)
    {
        try
        {
            _fdcCts?.Cancel();
            _fdcListener?.Stop();
            _fdcCts = null;
            _fdcListener = null;
            _fdcListenerTask = null;
        }
        catch
        {
            // Keep close flow robust even if background listener races.
        }
    }
}
