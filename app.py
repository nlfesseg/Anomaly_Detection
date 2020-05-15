import os
import time

import paho.mqtt.client as mqttClient
from flask import Flask, render_template, session, request
from flask_socketio import SocketIO, emit, join_room, leave_room
from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer

import util
from detector import Detector

async_mode = None

observers = {}

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

socketio = SocketIO(app, async_mode=async_mode)


@socketio.on('connect')
def connect():
    emit('my_response', {'data': 'Connected!'})


@socketio.on('connect_mqttclient')
def connect_mqttclient(args):
    broker_address = args['broker_address']
    port = int(args['port'])
    username = args['username']
    password = args['password']
    session['mqttclient'] = mqttClient.Client(clean_session=True)  # create new instance
    if username or password:
        session['mqttclient'].username_pw_set(username, password=password)
    try:
        session['mqttclient'].connect(broker_address, port=port)
        session['mqttclient'].loop_start()
        time.sleep(1)
        if session['mqttclient'].is_connected():
            emit('mqtt_connection_success_response', {'data': 'MQTT: Client is connected!'})
        else:
            emit('mqtt_connection_failed_response', {'data': 'MQTT: Bad connection! Invalid username or password'})
    except:
        emit('mqtt_connection_failed_response', {'data': 'MQTT: Bad connection! Broker address might be wrong'})


@socketio.on('disconnect_mqttclient')
def disconnect_mqttclient():
    session['mqttclient'].loop_stop()
    session['mqttclient'].disconnect()
    emit('mqtt_disconnect_response', {'data': 'MQTT: Client disconnected'})


@socketio.on('load_run_ids')
def load_run_ids():
    emit('run_ids_response', {'data': util.get_run_ids()})


@socketio.on('load_csv')
def load_csv(args):
    filename = args['filename']
    # ToDO: raise error
    session['detector'] = Detector(filename)
    emit('csv_response')


@socketio.on('load_run')
def load_run(args):
    run_id = args['run_id']
    # ToDO: raise error
    feature_ids = util.get_feature_ids(run_id)
    config = util.read_config(run_id)
    emit('run_response', {'feature_ids': feature_ids, 'config': config})


@socketio.on('delete_run')
def delete_run(args):
    run_id = args['run_id']
    util.delete_run('runs', run_id)
    load_run_ids()


@socketio.on('load_correlation_matrix')
def load_correlation_matrix():
    corr = abs(session['detector'].dataset.corr())
    emit('correlation_response', {'data': corr.values.tolist(), 'index': corr.index.values.tolist(),
                                  'columns': corr.columns.values.tolist()})


@socketio.on('train')
def train(args):
    config_dict = args
    config_dict = dict((k, int(v)) for k, v in config_dict.items())
    session['detector'].training = True
    if session['detector'].train(config_dict):
        emit('train_response')
    else:
        emit('training_stopped_response')


@socketio.on('stop_training')
def stop_training():
    session['detector'].training = False


@socketio.on('stop_running')
def stop_running():
    if 'observer_room' in session:
        leave_room(session['observer_room'])
        disconnect(current_session=session)
        del session['observer_room']
    emit('running_stopped_response')


@socketio.on('run')
def run(args):
    run_id = args['run_id']
    feature_ids = args['feature_ids']
    session['detector'].selected_feature_ids = feature_ids
    session['detector'].load(run_id)
    path = os.path.join('data', session['detector'].filename)
    if not path in observers:
        observers[path] = {}
        observers[path]['observer'] = Observer()
        patterns = [path]
        my_event_handler = PatternMatchingEventHandler(patterns, "", False, True)
        my_event_handler.on_modified = on_modified
        observers[path]['observer'].schedule(my_event_handler, path='data', recursive=False)
        observers[path]['observer'].start()
        observers[path]['old'] = 0
        observers[path]['count'] = 0
    session['observer_room'] = path
    observers[path]['count'] += 1
    join_room(session['observer_room'])


@socketio.on('predict')
def predict(args):
    if session['detector'].update():
        results = session['detector'].predict(int(args['view_history']), int(args['past_history']),
                                              int(args['future_target']))
        emit('result_response', {'data': results}, room=request.sid)


@socketio.on('disconnect')
def disconnect(current_session=None):
    if current_session is None:
        current_session = session
    if 'observer_room' in current_session:
        observers[current_session['observer_room']]['count'] -= 1
        if observers[current_session['observer_room']]['count'] < 1:
            observers[current_session['observer_room']]['observer'].stop()
            observers[current_session['observer_room']]['observer'].join()
            del observers[current_session['observer_room']]


def on_modified(event):
    global observers
    statbuf = os.stat(event.src_path)
    new = statbuf.st_mtime
    if (new - observers[event.src_path]['old']) > 0.5:
        socketio.emit('modified_event_response', room=event.src_path)
    observers[event.src_path]['old'] = new


@app.route('/')
def index():
    return render_template("index.html", csv_files=util.get_csv_filenames(),
                           runs=util.get_run_ids())


if __name__ == '__main__':
    socketio.run(app, debug=True)
