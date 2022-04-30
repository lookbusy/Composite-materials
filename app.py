from flask import Flask, request, render_template
from tensorflow import keras

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('main.html')


def matrix_processing(params):
    model = keras.models.load_model('E:/PyCharm/PyCharmProjects/TryFlask/models')
    predict = model.predict(params)
    return f'Соотношение матрица-наполнитель = {predict}'


@app.route('/matrix_predict/', methods=['post', 'get'])
def matrix_params():
    message = ''
    if request.method == 'POST':
        param1 = request.form.get('param1')
        param2 = request.form.get('param2')
        param3 = request.form.get('param3')
        param4 = request.form.get('param4')
        param5 = request.form.get('param5')
        param6 = request.form.get('param6')
        param7 = request.form.get('param7')
        param8 = request.form.get('param8')
        param9 = request.form.get('param9')
        param10 = request.form.get('param10')
        param11 = request.form.get('param11')
        param12 = request.form.get('param12')
        param1 = float(param1)
        param2 = float(param2)
        param3 = float(param3)
        param4 = float(param4)
        param5 = float(param5)
        param6 = float(param6)
        param7 = float(param7)
        param8 = float(param8)
        param9 = float(param9)
        param10 = float(param10)
        param11 = float(param11)
        param12 = float(param12)
        params = (param1, param2, param3, param4, param5, param6,
                  param7, param8, param9, param10, param11, param12)
        message = matrix_processing(params)
    return render_template('matrix_predict.html', message=message)


if __name__ == '__main__':
    app.run()
