using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Text;
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

    public Form1()
    {
        InitializeComponent();
        SetupGrid();
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
}
