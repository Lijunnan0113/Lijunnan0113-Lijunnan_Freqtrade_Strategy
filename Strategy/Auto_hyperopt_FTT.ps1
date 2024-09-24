# 设置.bat文件和.cfg文件的路径
$batFilePath1 = "  "
$batFilePath2 = "  "
$cfgFilePath = "  "
$downloadConfigBatPath = "  "

# 设置时间间隔
$checkInterval = 600  # 10分钟，单位为秒

# 定义一个循环，每天执行一次
while ($true) {
    # 获取当前时间和下次执行时间（设置为当天的凌晨2点）
    $now = Get-Date
    $nextRunTime = (Get-Date -Hour 1 -Minute 0 -Second 0)

    # 如果当前时间已经超过了今天的凌晨2点，则设置为明天的凌晨2点
    if ($now -ge $nextRunTime) {
        $nextRunTime = $nextRunTime.AddDays(1)
    }

    # 计算剩余时间（秒）
    $timeRemaining = ($nextRunTime - $now).TotalSeconds

    # 每10分钟提示一次，直到到达下次执行时间
    while ($timeRemaining -gt 0) {
        $hoursRemaining = [math]::Floor($timeRemaining / 3600)
        $minutesRemaining = [math]::Floor(($timeRemaining % 3600) / 60)
        Write-Output "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] 距离下次执行hyperopt还有 $hoursRemaining 小时 $minutesRemaining 分钟"
        Start-Sleep -Seconds $checkInterval
        $timeRemaining -= $checkInterval
    }

    # 获取当天和前4天的日期
    $today = Get-Date -Format "yyyyMMdd"
    $pastDays = (Get-Date).AddDays(-4).ToString("yyyyMMdd")
    $newTimerange = "--timerange=$pastDays-$today"
    # 输出修改后的时间范围
    Write-Output "修改的时间范围为: $newTimerange"

    # 修改第一个.bat文件的timerange参数
    $batFileContent1 = Get-Content $batFilePath1
    $updatedContent1 = $batFileContent1 -replace '--timerange=\d{8}-\d{8}', $newTimerange
    Set-Content $batFilePath1 $updatedContent1
    Write-Output "Updated first .bat file with new timerange: $newTimerange at $(Get-Date)"

    # 修改第二个.bat文件的timerange参数
    $batFileContent2 = Get-Content $batFilePath2
    $updatedContent2 = $batFileContent2 -replace '--timerange=\d{8}-\d{8}', $newTimerange
    Set-Content $batFilePath2 $updatedContent2
    Write-Output "Updated second .bat file with new timerange: $newTimerange at $(Get-Date)"

    # 冷却10秒
    Start-Sleep -Seconds 10

    # 执行下载配置数据的 .bat 文件
    Start-Process $downloadConfigBatPath -Wait
    Write-Output "下载配置数据的 .bat 文件已执行完成 at $(Get-Date)"

    # 执行第一个.bat文件
    Start-Process $batFilePath1 -Wait
    Write-Output "第一个 .bat 文件已执行完成 at $(Get-Date)"

    # 冷却10秒
    Start-Sleep -Seconds 10

    # 执行第二个.bat文件
    Start-Process $batFilePath2 -Wait
    Write-Output "第二个 .bat 文件已执行完成at $(Get-Date)"

    # 计算到下次执行的时间，并休眠
    $timeRemaining = ($nextRunTime.AddDays(1) - (Get-Date)).TotalSeconds
}
