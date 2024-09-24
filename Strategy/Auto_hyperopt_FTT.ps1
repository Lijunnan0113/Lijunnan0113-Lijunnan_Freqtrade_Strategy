# ����.bat�ļ���.cfg�ļ���·��
$batFilePath1 = "D:\WORK\Trading\freqtrade-2024.8\03.hyperopt_FBB_FTT_DWT_LongShort.bat"
$batFilePath2 = "D:\WORK\Trading\freqtrade-2024.8\03.hyperopt_FBB_FTT_DWT_LongShort_sell.bat"
$cfgFilePath = "D:\WORK\Trading\freqtrade-2024.8\user_data\strategies\FBB_FTT_DWT_LongShort.json"
$downloadConfigBatPath = "D:\WORK\Trading\freqtrade-2024.8\01-b����ָ��config����.bat"

# ����ʱ����
$checkInterval = 600  # 10���ӣ���λΪ��

# ����һ��ѭ����ÿ��ִ��һ��
while ($true) {
    # ��ȡ��ǰʱ����´�ִ��ʱ�䣨����Ϊ������賿2�㣩
    $now = Get-Date
    $nextRunTime = (Get-Date -Hour 1 -Minute 0 -Second 0)

    # �����ǰʱ���Ѿ������˽�����賿2�㣬������Ϊ������賿2��
    if ($now -ge $nextRunTime) {
        $nextRunTime = $nextRunTime.AddDays(1)
    }

    # ����ʣ��ʱ�䣨�룩
    $timeRemaining = ($nextRunTime - $now).TotalSeconds

    # ÿ10������ʾһ�Σ�ֱ�������´�ִ��ʱ��
    while ($timeRemaining -gt 0) {
        $hoursRemaining = [math]::Floor($timeRemaining / 3600)
        $minutesRemaining = [math]::Floor(($timeRemaining % 3600) / 60)
        Write-Output "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] �����´�ִ��hyperopt���� $hoursRemaining Сʱ $minutesRemaining ����"
        Start-Sleep -Seconds $checkInterval
        $timeRemaining -= $checkInterval
    }

    # ��ȡ�����ǰ4�������
    $today = Get-Date -Format "yyyyMMdd"
    $pastDays = (Get-Date).AddDays(-4).ToString("yyyyMMdd")
    $newTimerange = "--timerange=$pastDays-$today"
    # ����޸ĺ��ʱ�䷶Χ
    Write-Output "�޸ĵ�ʱ�䷶ΧΪ: $newTimerange"

    # �޸ĵ�һ��.bat�ļ���timerange����
    $batFileContent1 = Get-Content $batFilePath1
    $updatedContent1 = $batFileContent1 -replace '--timerange=\d{8}-\d{8}', $newTimerange
    Set-Content $batFilePath1 $updatedContent1
    Write-Output "Updated first .bat file with new timerange: $newTimerange at $(Get-Date)"

    # �޸ĵڶ���.bat�ļ���timerange����
    $batFileContent2 = Get-Content $batFilePath2
    $updatedContent2 = $batFileContent2 -replace '--timerange=\d{8}-\d{8}', $newTimerange
    Set-Content $batFilePath2 $updatedContent2
    Write-Output "Updated second .bat file with new timerange: $newTimerange at $(Get-Date)"

    # ��ȴ10��
    Start-Sleep -Seconds 10

    # ִ�������������ݵ� .bat �ļ�
    Start-Process $downloadConfigBatPath -Wait
    Write-Output "�����������ݵ� .bat �ļ���ִ����� at $(Get-Date)"

    # ִ�е�һ��.bat�ļ�
    Start-Process $batFilePath1 -Wait
    Write-Output "��һ�� .bat �ļ���ִ����� at $(Get-Date)"

    # ��ȴ10��
    Start-Sleep -Seconds 10

    # ִ�еڶ���.bat�ļ�
    Start-Process $batFilePath2 -Wait
    Write-Output "�ڶ��� .bat �ļ���ִ�����at $(Get-Date)"

    # ���㵽�´�ִ�е�ʱ�䣬������
    $timeRemaining = ($nextRunTime.AddDays(1) - (Get-Date)).TotalSeconds
}
