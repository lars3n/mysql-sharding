#coding:utf-8
import logging

import os

import time

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
from abc import abstractmethod, ABCMeta
import traceback
from optparse import OptionParser

try:
    import pymysql
except ImportError, e:
    pymysql = None
try:
    import MySQLdb
except ImportError, e:
    MySQLdb = None

logger.info('pymysql:%s'% str(pymysql))
logger.info('MySQLdb:%s'% str(MySQLdb))
assert pymysql is not None or MySQLdb is not None


class AbsDBCTX():
    __metaclass__ = ABCMeta

    def __init__(self, db_conf):
        self.pool = None
        self.conn = None
        self.cursor = None
        self.db_conf = db_conf

    @abstractmethod
    def prepare(self):
        pass

    @abstractmethod
    def finish(self):
        pass

    def __enter__(self):
        self.prepare()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        err = None
        try:
            # self.cursor.close()
            # self.conn.close()
            self.finish()
        except Exception, e:
            err = e
        if exc_val or exc_tb:
            raise Exception(traceback.format_exc())
        if err:
            raise Exception(str(err))


class MySQLdbDBCTX(AbsDBCTX):

    def prepare(self):
        db_conf = self.db_conf
        if pymysql:
            self.conn = pymysql.connect(
                host=db_conf['host'], port=db_conf['port'], user=db_conf['user'], passwd=db_conf['password'],
                db=db_conf['database'], charset=db_conf.get('charset', 'utf8'),
            )
            self.cursor = self.conn.cursor(
                cursor=pymysql.cursors.DictCursor
            )
        elif MySQLdb:
            self.conn = MySQLdb.connect(
                host=db_conf['host'], port=db_conf['port'], user=db_conf['user'],
                passwd=db_conf['password'],
                db=db_conf['database'], charset=db_conf.get('charset', 'utf8'),
            )
            self.cursor = self.conn.cursor(cursorclass=MySQLdb.cursors.DictCursor)

        return self

    def finish(self):
        try:
            self.cursor.close()
            self.conn.close()
        except Exception, e:
            logger.error(e)


class Main(object):
    def __init__(self, db_ctx):
        self.id_ = 0
        self.db_ctx = db_ctx

    def main(self, args, options):
        if options.opt == 'sharding':
            table_ori = args[0]
            sharding_limit = int(args[1])
            assert options.split > 0
            self.split = options.split
            self.sharding(table_ori, sharding_limit)

        elif options.opt == 'del_src':
            table_ori = args[0]
            start_id = int(args[1])
            end_id = int(args[2])
            with self.db_ctx as db_ctx_:
                split_num = options.split
                self.delete_src(db_ctx_, table_ori, start_id, end_id, split_num)

    def sharding(self, table_ori, sharding_limit):
        self.table_ori = table_ori = table_ori
        self.table_map = table_map = self.table_ori + '_meta'

        batch_limit = sharding_limit
        table_num = 0
        with self.db_ctx as db_ctx_:
            conn, cursor = db_ctx_.conn, db_ctx_.cursor

            sql = self.create_map_table(self.table_map)
            cursor.execute(sql)
            conn.commit()

            cursor.execute('select id from {table_ori} limit 0, 1'.format(table_ori=self.table_ori))
            first_id = cursor.fetchone()['id']
            cursor.execute('select id from {table_ori} order BY id desc limit 0,1'.format(table_ori=self.table_ori))
            last_id = cursor.fetchone()['id']
            logger.info(u'start:%d , end:%d' % (first_id, last_id))
            start = first_id

            total_count = 0

            if (last_id - first_id) < batch_limit:
                logger.error(u'未够数据拆分')
                return

            tables = self.get_tables(cursor)
            sharding_tables = []
            prefix = self.table_ori + '_'
            for table in tables:
                if table.startswith(prefix):
                    if table == self.table_map:
                        continue
                    num = table.lstrip(prefix)
                    try:
                        sharding_tables.append(int(num))
                    except:
                        pass

            if sharding_tables:
                sharding_tables.sort()
                table_num = sharding_tables[-1] + 1

            logger.info(u'开始切分的数据库编号:%d' % table_num)

            while 1:
                end = start+batch_limit
                qry_args = 'where {start} <= id and id < {end}'.format(start=start, end=end)
                logger.info(qry_args)

                sql = 'select count(id) count_a from {table_ori} {qry_args}'.format(table_ori=self.table_ori, qry_args=qry_args)
                cursor.execute(sql)
                count_a = cursor.fetchone()['count_a']
                if end > last_id:
                    break

                new_table = self.table_ori + '_%s' % str(table_num)

                sql = """
insert into {table_meta}(table_name, val0, val1) values('{table_name}', {start_num}, {limit_num}) ON DUPLICATE KEY UPDATE table_name='{table_name}',val0={start_num},val1={limit_num}                
                 """.format(table_meta=self.table_map, table_name=new_table, start_num=total_count,
                            limit_num=count_a)

                cursor.execute(sql)
                # conn.commit()

                total_count += count_a

                table_num += 1
                if new_table in tables:
                    start += batch_limit
                    continue

                self.fork_table(cursor, new_table)

                # sql = 'insert into {table_new} SELECT * FROM {table_ori} {qry_args}'
                # sql = sql.format(table_new=new_table, table_ori=self.table_ori, qry_args=qry_args)
                # cursor.execute(sql)
                # conn.commit()
                split_num = self.split

                self. move(db_ctx_, table_ori, new_table, start, end, split_num)

                logger.info(u'插入完毕, 开始删除原数据')

                self.delete_src(db_ctx_, table_ori, start, end, split_num)
                # sql = 'delete from {table_ori} {qry_args}'
                # sql = sql.format(table_ori=self.table_ori, qry_args=qry_args)
                # cursor.execute(sql)
                # conn.commit()

                start += batch_limit

            conn.commit()
            logger.info(u'全部结束')

    def move(self, db_ctx_, table_ori, new_table, start, end, split_num):
        next_num = start + split_num
        logger.info(u'start move id from %d to %d' % (start, end))
        while 1:
            if not self.check_load_balance(16):
                logger.warn(u'负载过高, 暂停删除')
                time.sleep(5)
                continue
            if next_num > end:
                break
            qry_args = ' where {start} <= id and id < {end} '.format(start=start, end=next_num)
            logger.info(u'moving ' + qry_args)
            sql = 'insert into {table_new} SELECT * FROM {table_ori} {qry_args}'
            sql = sql.format(table_new=new_table, table_ori=table_ori, qry_args=qry_args)
            db_ctx_.cursor.execute(sql)
            db_ctx_.conn.commit()

            start = next_num
            next_num = next_num + split_num
        logger.info(u'successful moved')

    def delete_src(self, db_ctx_, table_ori, start, end, split_num):
        # total_num = end - start
        next_num = start + split_num
        logger.info(u'start delete id from %d to %d' % (start, end))
        while 1:
            if not self.check_load_balance(16):
                logger.warn(u'负载过高, 暂停删除')
                time.sleep(5)
                continue
            if next_num > end:
                break
            qry_args = ' where {start} <= id and id < {end} '.format(start=start, end=next_num)
            logger.info(u'deleting ' + qry_args)
            sql = 'delete from {table_ori} {qry_args}'.format(table_ori=table_ori, qry_args=qry_args)
            logger.info(sql)
            db_ctx_.cursor.execute(sql)
            db_ctx_.conn.commit()

            start = next_num
            next_num = next_num + split_num

        logger.info(u'successful deleted')

    def get_tables(self, cursor=None):
        cursor.execute('select table_name from information_schema.tables where table_schema="{db_name}"'.format(db_name='poster'))
        reses = cursor.fetchall()
        return map(lambda x: x['table_name'], reses)

    def fork_table(self, cursor, table_name):
        sql = 'CREATE  TABLE IF NOT EXISTS {new_table} (LIKE {ori_table}); '
        # sql = 'CREATE  TABLE IF NOT EXISTS {new_table} SELECT {colums_ltr} FROM {ori_table} where ; '
        sql = sql.format(new_table=table_name, ori_table=self.table_ori)
        cursor.execute(sql)

        sql = 'ALTER TABLE {new_table} CHANGE COLUMN `id` `id` INT(10) UNSIGNED NOT NULL '.format(new_table=table_name)
        cursor.execute(sql)

    def create_map_table(self, table_name):
        sql ="""
CREATE TABLE IF NOT EXISTS `{table_name}` (
  `id` int(11) unsigned NOT NULL AUTO_INCREMENT,
  `table_name` varchar(45) NOT NULL,
  `val0` int(11) NOT NULL,
  `val1` int(11) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=8 DEFAULT CHARSET=utf8mb4            
        """.format(table_name=table_name)
        return sql

    def check_load_balance(self, max_balance):
        with os.popen('uptime') as f:
            res = f.read()
            data = res.split('load average:')[-1].strip().strip('\n').split(', ')
            if float(data[0]) > max_balance:
                return False
            return True


def init_opt_parser():
    parser = OptionParser()
    parser.add_option('--host', type='string', dest='host', default='127.0.0.1')
    parser.add_option('--port', type=int, action='store', dest='port', default=3306)
    parser.add_option('-u', '--user', type='string', dest='user', default='root')
    parser.add_option('-p', '--password', type='string', dest='password', default='')
    parser.add_option('-d', '--database', type='string', dest='database', default='')
    parser.add_option('-s', '--split', type=int, dest='split', default=1000, help=u'每一次操作的单位')
    parser.add_option('--opt', type='string', dest='opt', default='sharding', help=u'sharding/del_src')

    return parser


if __name__ == '__main__':
    parser = init_opt_parser()
    (options, args) = parser.parse_args()

    config = dict(
        host=options.host,
        port=options.port,
        database=options.database,
        user=options.user,
        password=options.password,
    )
    logger.info(config)
    db_ctx = MySQLdbDBCTX(config)

    try:
        Main(db_ctx).main(args, options)
    except Exception, e:
        logger.error(traceback.format_exc())




