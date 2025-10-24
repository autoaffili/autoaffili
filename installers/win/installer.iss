; Inno Setup script (installer.iss) - sketch
[Setup]
AppName=Autoaffili
AppVersion=0.1.0
DefaultDirName={localappdata}\autoaffili
DisableProgramGroupPage=yes
OutputBaseFilename=autoaffili-installer-0.1.0
; Request elevation via UAC so we can install the plugin globally
PrivilegesRequired=admin
PrivilegesRequiredOverridesAllowed=dialog
; Proactively close running OBS so DLL replacement succeeds
CloseApplications=force
; Ensure helper is closed so ProgramData files can be replaced
CloseApplicationsFilter=obs64.exe;obs32.exe;autoaffili.exe
RestartApplications=no
; RestartManagerSupport not supported by current compiler version; commented for compatibility
; RestartManagerSupport=yes

[Files]
; Install plugin DLL into global OBS plugins (Program Files)
Source: "..\\..\\obs-plugin\\build_x64\\rundir\\RelWithDebInfo\\autoaffili.dll"; DestDir: "{commonpf64}\\obs-studio\\obs-plugins\\64bit"; DestName: "autoaffili.dll"; Flags: ignoreversion restartreplace

; Install plugin metadata for Plugin Manager (global data path)
Source: "..\\..\\obs-plugin\\data\\obs-plugins\\autoaffili\\plugin.json"; DestDir: "{commonpf64}\\obs-studio\\data\\obs-plugins\\autoaffili"; Flags: ignoreversion
Source: "..\\..\\obs-plugin\\data\\locale\\en-US.ini"; DestDir: "{commonpf64}\\obs-studio\\data\\obs-plugins\\autoaffili\\locale"; Flags: ignoreversion

; Install service (helper) into ProgramData; plugin will prefer launching from here
; Ensure helper binaries can be replaced without reboot; ignoreversion avoids version checks on packed EXE files
Source: "..\..\service\dist\autoaffili\*"; DestDir: "{commonappdata}\\autoaffili"; Flags: recursesubdirs createallsubdirs ignoreversion

[InstallDelete]
; Fully replace ProgramData helper on each install to avoid stale bundles
Type: filesandordirs; Name: "{commonappdata}\\autoaffili"

[Run]
; No run. OBS plugin will launch the service when OBS starts.

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

; Signing (enable for release)
; To enable signing, uncomment the two lines below and provide certificate defines
; in your build environment (e.g., ISCC /DCertFile=path\\to\\cert.pfx /DCertPass=secret)
;[Setup]
;SignTool=SignCmd
;[SignTool]
;Name: "SignCmd"; Command: "signtool sign /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 /f {#CertFile} /p {#CertPass} $f"

[Code]
procedure KillHelper();
var
  ResultCode: Integer;
begin
  Exec('taskkill.exe', '/IM autoaffili.exe /F', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
end;
