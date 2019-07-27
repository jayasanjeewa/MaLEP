from flask import Flask, render_template,request,json
from bottle import route, run, Bottle
from bottle_pymysql import Plugin
import pymysql.cursors

app = Bottle()

plugin = Plugin(dbuser='eurecatool', dbpass='LgYTv*@pY7v7w7RvX?qTq2pAqNykL+', dbname='eureca')
app.install(plugin)

@route('/snapshot')
def show():
   # c = plugin.cursor()
   # c.execute('SELECT * from Snapshot where id=1')



    connection = pymysql.connect(host='localhost',
                             user='eurecatool',
                             password='LgYTv*@pY7v7w7RvX?qTq2pAqNykL+',
                             db='eureca',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)


    app = Flask(__name__)

    output = '<h1>'

    with connection.cursor() as cursor:
      # Read a single record
      sql = "SELECT * from Snapshot"
      cursor.execute(sql)

      result = cursor.fetchall()

      for row in result:
          output += '</p>' + str(row['id']) +  '\t' + str(row['name']) +  '</p>';
      output += '</h1>'

      connection.commit();


    return output


    return "<h1>Hello World!</h1>"

run(host='0.0.0.0', port=8095)
