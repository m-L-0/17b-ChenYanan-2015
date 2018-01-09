import os
from flask import Flask, render_template, request, redirect, url_for
from flask_bootstrap import Bootstrap  
from validation import validation

UPLOAD_FOLDER = r'app\upload'
app = Flask(__name__)
bootstrap = Bootstrap(app)


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index_c_1.html')


@app.route('/upload_file', methods=['GET', 'POST'])  
def upload_file():
    if request.method == 'POST':
        request.files['file-zh-TW[]'].save('./image_test.jpg')
        rst = validation('./image_test.jpg')
        # file = request.files['file']  
        # print(file.name)
        return "{\"extra\": \""+rst+"\"}"
    return "未上传图片"
    # return render_template('upload.html', extra=rst)


# @app.route('/upload_file', methods=['GET', 'POST']) 
# def upload_file():
#     if request.method == 'POST':
#         file = request.files['file']
#         if file:
#             file.save(os.path.join(UPLOAD_FOLDER, file.filename))  
#             return redirect(url_for('.uploaded_file', filename=file.filename))       #跳转到预览页面  
#         return '<p> 你上传了不允许的文件类型 </p>'  
#     return 