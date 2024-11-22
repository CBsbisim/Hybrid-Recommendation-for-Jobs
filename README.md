# HybridRecommendation
# 职位混合推荐算法
运行文件执行
python3 /data/python/reco_api/api_reco.py

# 混合推荐算法思路：
由招聘信息根据用户评分的协同过滤方式进行排序推荐
结合招聘信息特征相似度
每天4点会进行自动保存一份推荐数据，存在/data/python/reco_api/date_loggings/目录下，名为时间戳.csv

# 接口调用：
端口为10002，通过访问：
localhost:10002/recommend?user_id= 进行获取
返回值为position_id

# 日志删除说明：
日志文件（保存错误文件）目前删除周期为7天
csv文件（保存推荐值）目前删除周期为30天
--执行文件--
auto-del-ago-log.sh
已写入crontab -e #系统定时任务的配置信息


# 服务器上执行
nohup python3 -u api_reco.py > outinfo.txt 2>&1 &
# 输入exit后再关闭终端
exit

# 查看资源占用情况
top -p 进程号
# 查询进程号pid
lsof -i:10002
