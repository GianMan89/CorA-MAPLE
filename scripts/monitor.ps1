param(
    [string]$Workspace = "runs/tep_cv10x5_cleanft_v2_1",
    [string]$Study = "primary",
    [int]$Seconds = 60
)
if ($Seconds -lt 1) { throw "Seconds must be positive" }
while ($true) {
    Clear-Host
    Get-Date
    python -m cora_maple status --workspace $Workspace --study $Study
    Start-Sleep -Seconds $Seconds
}
