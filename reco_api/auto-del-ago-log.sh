#!/bin/sh
# 自动删除日志文件（7天）  删除csv模型缓存文件（30天）

#1、添加文件可运行权限
# chmod +x /data/python/reco_api/auto-del-ago-log.sh
#2、打开系统 定时任务的配置
# crontab -e
#3、添加配置 每天0:00执行任务
# 0 0 * * * /data/python/reco_api/auto-del-ago-log.sh
cd /data/python/reco_api
DIR='/data/python/reco_api'
source /etc/profile
nohup python3 -u api_reco.py > outinfo.txt 2>&1 &

echo "删除7天前的log日志" /data/python/reco_api/error_logs/
find /data/python/reco_api/error_logs/ -mtime +7 -name "*.log" -exec rm -rf {} \;

echo "删除30天前的csv文件" $DIR/date_loggings/
find /data/python/reco_api/date_loggings/ -mtime +30 -name "*.csv" -exec rm -rf {} \;







