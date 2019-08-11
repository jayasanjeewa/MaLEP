#!../bin/python
from flask import Flask, jsonify, json
from flask import abort
from flask import request

app = Flask(__name__)

tasks = [
    {
        'id': 1,
        'title': u'Buy groceries',
        'description': u'Milk, Cheese, Pizza, Fruit, Tylenol', 
        'done': False
    },
    {
        'id': 2,
        'title': u'Learn Python',
        'description': u'Need to find a good Python tutorial on the web', 
        'done': False
    }
]

@app.route('/todo/api/v1.0/tasks', methods=['GET'])
def get_tasks():
    return jsonify({'tasks': tasks})

@app.route('/todo/api/v1.0/tasks/<int:task_id>', methods=['GET'])
def get_task(task_id):
    task = [task for task in tasks if task['id'] == task_id]
    if len(task) == 0:
        abort(404)
    return jsonify({'task': task[0]})



@app.route('/malep/serverenergy/predict', methods=['POST'])
def get_predicted_energy():
    if not request.json :
        abort(400)

    processor_speed = request.json['processor_speed']
    no_of_cores = request.json['no_of_cores']
    memory = request.json['memory']
    reference_age_months = request.json['reference_age_months']

    predicted_utilized_energy = predict_utilized_power(processor_speed, no_of_cores, memory,
                                                                        reference_age_months)

    predicted_idle_energy = predict_idle_power(processor_speed, no_of_cores, memory,
                                              reference_age_months)


    predicted_consumption = {
        'utilized_power_consumption': json.dumps(predicted_utilized_energy),
        'idle_power_consumption': json.dumps(predicted_idle_energy)
    }

    return jsonify({'prediction': predicted_consumption}), 201

def predict_utilized_power(processor_speed, no_of_cores, memory, reference_age_months):
    return 514

def predict_idle_power(processor_speed, no_of_cores, memory, reference_age_months):
    return 142

@app.route('/todo/api/v1.0/tasks/<int:task_id>', methods=['PUT'])
def update_task(task_id):
    task = [task for task in tasks if task['id'] == task_id]
    if len(task) == 0:
        abort(404)
    if not request.json:
        abort(400)
    if 'title' in request.json and type(request.json['title']) != unicode:
        abort(400)
    if 'description' in request.json and type(request.json['description']) is not unicode:
        abort(400)
    if 'done' in request.json and type(request.json['done']) is not bool:
        abort(400)
    task[0]['title'] = request.json.get('title', task[0]['title'])
    task[0]['description'] = request.json.get('description', task[0]['description'])
    task[0]['done'] = request.json.get('done', task[0]['done'])
    return jsonify({'task': task[0]})

@app.route('/todo/api/v1.0/tasks/<int:task_id>', methods=['DELETE'])
def delete_task(task_id):
    task = [task for task in tasks if task['id'] == task_id]
    if len(task) == 0:
        abort(404)
    tasks.remove(task[0])
    return jsonify({'result': True})

if __name__ == '__main__':
    app.run(debug=True)

#test
#curl -i -H "Content-Type: application/json" -X POST -d '{"processor_speed":"100"}' http://localhost:5000/malep/serverenergy/predict
#chttps://blog.miguelgrinberg.com/post/designing-a-restful-api-with-python-and-flask


