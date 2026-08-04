namespace ReconSimCS;

partial class Form1
{
    private System.ComponentModel.IContainer components = null;

    protected override void Dispose(bool disposing)
    {
        if (disposing && (components != null))
        {
            components.Dispose();
        }
        base.Dispose(disposing);
    }

    private void InitializeComponent()
    {
        this.grpSettings = new System.Windows.Forms.GroupBox();
        this.lblIP = new System.Windows.Forms.Label();
        this.txtIP = new System.Windows.Forms.TextBox();
        this.lblPort = new System.Windows.Forms.Label();
        this.numPort = new System.Windows.Forms.NumericUpDown();
        this.lblPath = new System.Windows.Forms.Label();
        this.txtPath = new System.Windows.Forms.TextBox();
        this.btnBrowseFile = new System.Windows.Forms.Button();
        this.btnBrowseFolder = new System.Windows.Forms.Button();
        this.lblCounter = new System.Windows.Forms.Label();
        this.numCounter = new System.Windows.Forms.NumericUpDown();
        this.chkFlag0 = new System.Windows.Forms.CheckBox();
        this.chkFlag1 = new System.Windows.Forms.CheckBox();
        this.chkFlag2 = new System.Windows.Forms.CheckBox();
        this.chkFlag3 = new System.Windows.Forms.CheckBox();
        this.btnSend = new System.Windows.Forms.Button();
        this.lblFdcTitle = new System.Windows.Forms.Label();
        this.lblFdcPort = new System.Windows.Forms.Label();
        this.numFdcPort = new System.Windows.Forms.NumericUpDown();
        this.btnFdcListen = new System.Windows.Forms.Button();
        this.lblFdcPackets = new System.Windows.Forms.Label();
        this.gridResults = new System.Windows.Forms.DataGridView();
        this.txtLog = new System.Windows.Forms.RichTextBox();
        this.statusStrip = new System.Windows.Forms.StatusStrip();
        this.lblStatus = new System.Windows.Forms.ToolStripStatusLabel();

        this.grpSettings.SuspendLayout();
        ((System.ComponentModel.ISupportInitialize)(this.numPort)).BeginInit();
        ((System.ComponentModel.ISupportInitialize)(this.numCounter)).BeginInit();
        ((System.ComponentModel.ISupportInitialize)(this.numFdcPort)).BeginInit();
        ((System.ComponentModel.ISupportInitialize)(this.gridResults)).BeginInit();
        this.statusStrip.SuspendLayout();
        this.SuspendLayout();

        // 
        // grpSettings
        // 
        this.grpSettings.Controls.Add(this.lblIP);
        this.grpSettings.Controls.Add(this.txtIP);
        this.grpSettings.Controls.Add(this.lblPort);
        this.grpSettings.Controls.Add(this.numPort);
        this.grpSettings.Controls.Add(this.lblPath);
        this.grpSettings.Controls.Add(this.txtPath);
        this.grpSettings.Controls.Add(this.btnBrowseFile);
        this.grpSettings.Controls.Add(this.btnBrowseFolder);
        this.grpSettings.Controls.Add(this.lblCounter);
        this.grpSettings.Controls.Add(this.numCounter);
        this.grpSettings.Controls.Add(this.chkFlag0);
        this.grpSettings.Controls.Add(this.chkFlag1);
        this.grpSettings.Controls.Add(this.chkFlag2);
        this.grpSettings.Controls.Add(this.chkFlag3);
        this.grpSettings.Controls.Add(this.btnSend);
        this.grpSettings.Controls.Add(this.lblFdcTitle);
        this.grpSettings.Controls.Add(this.lblFdcPort);
        this.grpSettings.Controls.Add(this.numFdcPort);
        this.grpSettings.Controls.Add(this.btnFdcListen);
        this.grpSettings.Controls.Add(this.lblFdcPackets);
        this.grpSettings.Location = new System.Drawing.Point(12, 12);
        this.grpSettings.Name = "grpSettings";
        this.grpSettings.Size = new System.Drawing.Size(960, 285);
        this.grpSettings.TabIndex = 0;
        this.grpSettings.TabStop = false;
        this.grpSettings.Text = "Target Inspection PC & Input Volume Settings";

        // lblIP
        this.lblIP.Location = new System.Drawing.Point(15, 30);
        this.lblIP.Size = new System.Drawing.Size(130, 23);
        this.lblIP.Text = "Inspection PC IP:";

        // txtIP
        this.txtIP.Location = new System.Drawing.Point(150, 27);
        this.txtIP.Size = new System.Drawing.Size(200, 27);
        this.txtIP.Text = "192.168.1.133";

        // lblPort
        this.lblPort.Location = new System.Drawing.Point(370, 30);
        this.lblPort.Size = new System.Drawing.Size(50, 23);
        this.lblPort.Text = "Port:";

        // numPort
        this.numPort.Location = new System.Drawing.Point(420, 27);
        this.numPort.Maximum = new decimal(new int[] { 65535, 0, 0, 0 });
        this.numPort.Minimum = new decimal(new int[] { 1, 0, 0, 0 });
        this.numPort.Size = new System.Drawing.Size(100, 27);
        this.numPort.Value = new decimal(new int[] { 8000, 0, 0, 0 });

        // lblPath
        this.lblPath.Location = new System.Drawing.Point(15, 70);
        this.lblPath.Size = new System.Drawing.Size(130, 23);
        this.lblPath.Text = "Volume File/Folder:";

        // txtPath
        this.txtPath.Location = new System.Drawing.Point(150, 67);
        this.txtPath.Size = new System.Drawing.Size(560, 27);
        this.txtPath.Text = @"Z:\volume_compensated.tif";

        // btnBrowseFile
        this.btnBrowseFile.Location = new System.Drawing.Point(720, 65);
        this.btnBrowseFile.Size = new System.Drawing.Size(110, 30);
        this.btnBrowseFile.Text = "Browse File...";
        this.btnBrowseFile.Click += new System.EventHandler(this.btnBrowseFile_Click);

        // btnBrowseFolder
        this.btnBrowseFolder.Location = new System.Drawing.Point(835, 65);
        this.btnBrowseFolder.Size = new System.Drawing.Size(115, 30);
        this.btnBrowseFolder.Text = "Browse Folder...";
        this.btnBrowseFolder.Click += new System.EventHandler(this.btnBrowseFolder_Click);

        // lblCounter
        this.lblCounter.Location = new System.Drawing.Point(15, 110);
        this.lblCounter.Size = new System.Drawing.Size(130, 23);
        this.lblCounter.Text = "Packet Counter:";

        // numCounter
        this.numCounter.Location = new System.Drawing.Point(150, 107);
        this.numCounter.Maximum = new decimal(new int[] { 999999, 0, 0, 0 });
        this.numCounter.Minimum = new decimal(new int[] { 1, 0, 0, 0 });
        this.numCounter.Size = new System.Drawing.Size(120, 27);
        this.numCounter.Value = new decimal(new int[] { 1, 0, 0, 0 });

        // chkFlags
        this.chkFlag0.Location = new System.Drawing.Point(300, 108);
        this.chkFlag0.Size = new System.Drawing.Size(120, 24);
        this.chkFlag0.Text = "Flag 0 (Enable)";
        this.chkFlag0.Checked = true;

        this.chkFlag1.Location = new System.Drawing.Point(430, 108);
        this.chkFlag1.Size = new System.Drawing.Size(80, 24);
        this.chkFlag1.Text = "Flag 1";

        this.chkFlag2.Location = new System.Drawing.Point(520, 108);
        this.chkFlag2.Size = new System.Drawing.Size(80, 24);
        this.chkFlag2.Text = "Flag 2";

        this.chkFlag3.Location = new System.Drawing.Point(610, 108);
        this.chkFlag3.Size = new System.Drawing.Size(80, 24);
        this.chkFlag3.Text = "Flag 3";

        // btnSend
        this.btnSend.BackColor = System.Drawing.Color.FromArgb(0, 180, 100);
        this.btnSend.ForeColor = System.Drawing.Color.White;
        this.btnSend.Font = new System.Drawing.Font("Segoe UI", 10F, System.Drawing.FontStyle.Bold);
        this.btnSend.Location = new System.Drawing.Point(150, 150);
        this.btnSend.Size = new System.Drawing.Size(800, 50);
        this.btnSend.Text = "SEND TCP TRIGGER & BENCHMARK SPEED";
        this.btnSend.UseVisualStyleBackColor = false;
        this.btnSend.Click += new System.EventHandler(this.btnSend_Click);

        // lblFdcTitle
        this.lblFdcTitle.Location = new System.Drawing.Point(15, 214);
        this.lblFdcTitle.Size = new System.Drawing.Size(300, 23);
        this.lblFdcTitle.Text = "FDC Monitor Receiver (from Inno3D):";

        // lblFdcPort
        this.lblFdcPort.Location = new System.Drawing.Point(15, 245);
        this.lblFdcPort.Size = new System.Drawing.Size(130, 23);
        this.lblFdcPort.Text = "Listen Port:";

        // numFdcPort
        this.numFdcPort.Location = new System.Drawing.Point(150, 242);
        this.numFdcPort.Maximum = new decimal(new int[] { 65535, 0, 0, 0 });
        this.numFdcPort.Minimum = new decimal(new int[] { 1, 0, 0, 0 });
        this.numFdcPort.Size = new System.Drawing.Size(110, 27);
        this.numFdcPort.Value = new decimal(new int[] { 8100, 0, 0, 0 });

        // btnFdcListen
        this.btnFdcListen.BackColor = System.Drawing.Color.FromArgb(36, 122, 213);
        this.btnFdcListen.ForeColor = System.Drawing.Color.White;
        this.btnFdcListen.Font = new System.Drawing.Font("Segoe UI", 9F, System.Drawing.FontStyle.Bold);
        this.btnFdcListen.Location = new System.Drawing.Point(280, 239);
        this.btnFdcListen.Size = new System.Drawing.Size(210, 32);
        this.btnFdcListen.Text = "Start FDC Listener";
        this.btnFdcListen.UseVisualStyleBackColor = false;
        this.btnFdcListen.Click += new System.EventHandler(this.btnFdcListen_Click);

        // lblFdcPackets
        this.lblFdcPackets.Location = new System.Drawing.Point(510, 245);
        this.lblFdcPackets.Size = new System.Drawing.Size(220, 23);
        this.lblFdcPackets.Text = "Packets: 0";

        // gridResults
        this.gridResults.AllowUserToAddRows = false;
        this.gridResults.AutoSizeColumnsMode = System.Windows.Forms.DataGridViewAutoSizeColumnsMode.Fill;
        this.gridResults.Location = new System.Drawing.Point(12, 310);
        this.gridResults.Name = "gridResults";
        this.gridResults.Size = new System.Drawing.Size(960, 200);
        this.gridResults.TabIndex = 1;

        // txtLog
        this.txtLog.BackColor = System.Drawing.Color.FromArgb(18, 18, 24);
        this.txtLog.ForeColor = System.Drawing.Color.LightGray;
        this.txtLog.Font = new System.Drawing.Font("Consolas", 9.5F);
        this.txtLog.Location = new System.Drawing.Point(12, 520);
        this.txtLog.Name = "txtLog";
        this.txtLog.ReadOnly = true;
        this.txtLog.Size = new System.Drawing.Size(960, 200);
        this.txtLog.TabIndex = 2;
        this.txtLog.Text = "";

        // statusStrip
        this.lblStatus.Text = "Ready to transmit volume trigger.";
        this.statusStrip.Items.Add(this.lblStatus);
        this.statusStrip.Location = new System.Drawing.Point(0, 725);
        this.statusStrip.Size = new System.Drawing.Size(984, 26);

        // Form1
        this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 20F);
        this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
        this.ClientSize = new System.Drawing.Size(984, 751);
        this.Controls.Add(this.grpSettings);
        this.Controls.Add(this.gridResults);
        this.Controls.Add(this.txtLog);
        this.Controls.Add(this.statusStrip);
        this.Name = "Form1";
        this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
        this.Text = "Recon PC - Volume TCP Sender & Latency Tester (C# .NET 8)";
        this.grpSettings.ResumeLayout(false);
        this.grpSettings.PerformLayout();
        ((System.ComponentModel.ISupportInitialize)(this.numPort)).EndInit();
        ((System.ComponentModel.ISupportInitialize)(this.numCounter)).EndInit();
        ((System.ComponentModel.ISupportInitialize)(this.numFdcPort)).EndInit();
        ((System.ComponentModel.ISupportInitialize)(this.gridResults)).EndInit();
        this.statusStrip.ResumeLayout(false);
        this.statusStrip.PerformLayout();
        this.ResumeLayout(false);
        this.PerformLayout();
    }

    private System.Windows.Forms.GroupBox grpSettings;
    private System.Windows.Forms.Label lblIP;
    private System.Windows.Forms.TextBox txtIP;
    private System.Windows.Forms.Label lblPort;
    private System.Windows.Forms.NumericUpDown numPort;
    private System.Windows.Forms.Label lblPath;
    private System.Windows.Forms.TextBox txtPath;
    private System.Windows.Forms.Button btnBrowseFile;
    private System.Windows.Forms.Button btnBrowseFolder;
    private System.Windows.Forms.Label lblCounter;
    private System.Windows.Forms.NumericUpDown numCounter;
    private System.Windows.Forms.CheckBox chkFlag0;
    private System.Windows.Forms.CheckBox chkFlag1;
    private System.Windows.Forms.CheckBox chkFlag2;
    private System.Windows.Forms.CheckBox chkFlag3;
    private System.Windows.Forms.Button btnSend;
    private System.Windows.Forms.Label lblFdcTitle;
    private System.Windows.Forms.Label lblFdcPort;
    private System.Windows.Forms.NumericUpDown numFdcPort;
    private System.Windows.Forms.Button btnFdcListen;
    private System.Windows.Forms.Label lblFdcPackets;
    private System.Windows.Forms.DataGridView gridResults;
    private System.Windows.Forms.RichTextBox txtLog;
    private System.Windows.Forms.StatusStrip statusStrip;
    private System.Windows.Forms.ToolStripStatusLabel lblStatus;
}
