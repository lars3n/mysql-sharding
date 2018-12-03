# mysql-sharding
MySQL 大表水平切分脚本

# 说明
```
以id 为切分标准,适合冷热数据分离较大的表,当系统负载过高时自动暂停切分,未做业务层实现
```

# 运行
* 切分
  `python sharding.py -u USERNAME -p PASSWORD -d DATABASE TABLE 500000`
  以500000 为单位将表切分成 TABLE_0, TABLE_1, TABLE_2 ...,TABLE 中保留切分完的数据

