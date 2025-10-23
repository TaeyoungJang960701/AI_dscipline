from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return 'Hello 김성시'

@app.route('/first', methods = ['GET'])
def first():
    name = request.args.get("name")
    return f"<h1>hello<span style = 'color:blue'>{name}</span></h1>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

