$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$ledgerRoot = Join-Path $repoRoot 'data/predictions'
$rows = @()
$batches = @()
$eastern = [TimeZoneInfo]::FindSystemTimeZoneById('Eastern Standard Time')
foreach ($file in Get-ChildItem -LiteralPath $ledgerRoot -Filter '*.json' | Sort-Object Name) {
    if ($file.Name.EndsWith('.results.json')) { continue }
    $batch = Get-Content -LiteralPath $file.FullName -Raw | ConvertFrom-Json
    $published = [DateTimeOffset]::Parse($batch.created_at)
    $settled = @{}
    $events = @()
    $legacy = Join-Path $ledgerRoot ($batch.run_id + '.results.json')
    if (Test-Path -LiteralPath $legacy) { $events += Get-Content -LiteralPath $legacy -Raw | ConvertFrom-Json }
    $eventDir = Join-Path $ledgerRoot ('settlements/' + $batch.run_id)
    if (Test-Path -LiteralPath $eventDir) {
        foreach ($eventFile in Get-ChildItem -LiteralPath $eventDir -Filter '*.json' | Sort-Object Name) {
            $events += Get-Content -LiteralPath $eventFile.FullName -Raw | ConvertFrom-Json
        }
    }
    foreach ($event in $events) { foreach ($result in $event.results) { $settled[$result.game_id] = $result } }
    $batches += [PSCustomObject]@{run=$batch.run_id;created=$batch.created_at;cutoff=$batch.data_cutoff;week=$batch.metadata.week;games=@($batch.predictions).Count;settled=$settled.Count;calibrated=@($batch.predictions | Where-Object { $_.preseason_calibration.weight -gt 0 }).Count;market=@($batch.predictions | Where-Object { $_.market_consensus }).Count;injury_week=$batch.metadata.injury_available_week;injury_stale=$batch.metadata.injury_stale_for_prediction_week}
    foreach ($p in $batch.predictions) {
        if ($p.commence_time) { $kickoff = [DateTimeOffset]::Parse($p.commence_time) }
        elseif ($p.gameday -and $p.gametime -and $p.gametime -ne 'TBD') {
            $local = [DateTime]::SpecifyKind([DateTime]::Parse($p.gameday + 'T' + $p.gametime), [DateTimeKind]::Unspecified)
            $kickoff = [DateTimeOffset][TimeZoneInfo]::ConvertTimeToUtc($local, $eastern)
        } else { continue }
        if ($published -ge $kickoff) { continue }
        $actual = $settled[$p.game_id]
        $independent = $p.predicted_home_margin
        if ($p.football_only) { $independent = $p.football_only.home_margin }
        elseif ($p.preseason_calibration.weight -gt 0) { $independent = $null }
        $marketMargin = $null
        $marketTotal = $null
        if ($p.market_consensus.snapshot_at) {
            $captured = [DateTimeOffset]::Parse($p.market_consensus.snapshot_at)
            if ($captured -le $published -and ($published - $captured).TotalDays -le 8) {
                $marketMargin = $p.market_consensus.spread.market_home_margin
                $marketTotal = $p.market_consensus.total.total
            }
        }
        $rows += [PSCustomObject]@{game=$p.game_id;week=$p.week;run=$batch.run_id;published=$published;lead_hours=($kickoff-$published).TotalHours;margin=$p.predicted_home_margin;independent=$independent;total=$p.total;actual_margin=$actual.actual_home_margin;actual_total=$actual.actual_total;market_margin=$marketMargin;market_total=$marketTotal;calibrated=($p.preseason_calibration.weight -gt 0)}
    }
}
function Get-Metrics($selected) {
    foreach ($week in @('all',1,2,3)) {
        $group = @($selected | Where-Object { $null -ne $_.actual_margin -and ($week -eq 'all' -or $_.week -eq $week) })
        if (-not $group.Count) { continue }
        $matched = @($group | Where-Object { $null -ne $_.market_margin -and $null -ne $_.market_total })
        $independent = @($group | Where-Object { $null -ne $_.independent })
        [PSCustomObject]@{
            week=$week;games=$group.Count;calibrated=@($group | Where-Object calibrated).Count
            margin_mae=($group | ForEach-Object { [math]::Abs($_.margin-$_.actual_margin) } | Measure-Object -Average).Average
            independent_games=$independent.Count
            independent_mae=($independent | ForEach-Object { [math]::Abs($_.independent-$_.actual_margin) } | Measure-Object -Average).Average
            total_mae=($group | ForEach-Object { [math]::Abs($_.total-$_.actual_total) } | Measure-Object -Average).Average
            matched=$matched.Count
            matched_margin_mae=($matched | ForEach-Object { [math]::Abs($_.margin-$_.actual_margin) } | Measure-Object -Average).Average
            market_margin_mae=($matched | ForEach-Object { [math]::Abs($_.market_margin-$_.actual_margin) } | Measure-Object -Average).Average
            market_total_mae=($matched | ForEach-Object { [math]::Abs($_.market_total-$_.actual_total) } | Measure-Object -Average).Average
            matched_total_mae=($matched | ForEach-Object { [math]::Abs($_.total-$_.actual_total) } | Measure-Object -Average).Average
        }
    }
}
$first = @($rows | Sort-Object published,run | Group-Object game | ForEach-Object { $_.Group | Select-Object -First 1 })
$latest = @($rows | Where-Object { $_.lead_hours -ge 1 } | Sort-Object published,run | Group-Object game | ForEach-Object { $_.Group | Select-Object -Last 1 })
$summary = [PSCustomObject]@{scope='Checked-in NFL ledger only; no new outcomes fetched';batches=$batches;first_metrics=@(Get-Metrics $first);latest_60m_metrics=@(Get-Metrics $latest)}
$summary | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'ledger-summary.json') -Encoding UTF8
$first | Export-Csv -LiteralPath (Join-Path $PSScriptRoot 'first-forecasts.csv') -NoTypeInformation
$latest | Export-Csv -LiteralPath (Join-Path $PSScriptRoot 'latest-60m-forecasts.csv') -NoTypeInformation
$summary | ConvertTo-Json -Depth 7
