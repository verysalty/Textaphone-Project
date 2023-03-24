from datetime import datetime

from bson.json_util import dumps
from flask import Flask, render_template, request, redirect, url_for
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from flask_socketio import SocketIO, join_room, leave_room
from pymongo.errors import DuplicateKeyError


from db import get_user, save_user, save_room, add_room_members, get_rooms_for_user, get_mode_for_user, get_room, is_room_member, \
    get_room_members, is_room_admin, update_room, remove_room_members, save_message, get_messages

#ml imports
import numpy as np  
from tensorflow.keras.models import load_model
import joblib
import json

# ML SENTIMENT ANALYSIS
def sentences_to_indices(X, word_to_index, max_len):
    m = X.shape[0]                         
    X_indices = np.zeros((m,max_len))
    for i in range(m):                               
        sentence_words = [i.lower() for i in X[i].split()]
        j = 0
        for w in sentence_words:
          try:
            X_indices[i, j] = word_to_index[w]
            j = j+1
          except:
            continue
      
    
    return X_indices

def label_to_emoji(label):
    return emoji.emojize(emoji_dictionary[str(label)], use_aliases=True)


model = load_model("my_model")

with open('word_to_index.json') as f:
	    word_to_index = json.load(f)



app = Flask(__name__)
app.secret_key = "sfdjkafnk"
socketio = SocketIO(app)
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)



@app.route('/')
def home():
    rooms = []
    if current_user.is_authenticated:
        usermode = get_mode_for_user(current_user.username)
        rooms = get_rooms_for_user(current_user.username)
        return render_template("index.html", rooms=rooms, usermode=usermode)
    if not current_user.is_authenticated:
        return render_template("login.html")

    
    


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    message = ''
    if request.method == 'POST':
        username = request.form.get('username')
        password_input = request.form.get('password')
        user = get_user(username)

        if user and user.check_password(password_input):
            login_user(user)
            return redirect(url_for('home'))
        else:
            message = 'Failed to login!'
    return render_template('login.html', message=message)


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    message = ''
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        mode = request.form.get('mode')

        try:
            save_user(username, email, password, mode)
            return redirect(url_for('login'))
        except DuplicateKeyError:
            message = "User already exists!"
    return render_template('signup.html', message=message)


@app.route("/logout/")
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))


@app.route('/create-room/', methods=['GET', 'POST'])
@login_required
def create_room():
    message = ''
    if request.method == 'POST':
        room_name = request.form.get('room_name')
        usernames = [username.strip() for username in request.form.get('members').split(',')]

        if len(room_name) and len(usernames):
            room_id = save_room(room_name, current_user.username)
            if current_user.username in usernames:
                usernames.remove(current_user.username)
            add_room_members(room_id, room_name, usernames, current_user.username)
            usermode = get_mode_for_user(current_user.username)
            return redirect(url_for('view_room', room_id=room_id, usermode=usermode))
        else:
            message = "Failed to create room"
    return render_template('create_room.html', message=message)


@app.route('/rooms/<room_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_room(room_id):
    room = get_room(room_id)
    if room and is_room_admin(room_id, current_user.username):
        existing_room_members = [member['_id']['username'] for member in get_room_members(room_id)]
        room_members_str = ",".join(existing_room_members)
        message = ''
        if request.method == 'POST':
            room_name = request.form.get('room_name')
            room['name'] = room_name
            update_room(room_id, room_name)

            new_members = [username.strip() for username in request.form.get('members').split(',')]
            members_to_add = list(set(new_members) - set(existing_room_members))
            members_to_remove = list(set(existing_room_members) - set(new_members))
            if len(members_to_add):
                add_room_members(room_id, room_name, members_to_add, current_user.username)
            if len(members_to_remove):
                remove_room_members(room_id, members_to_remove)
            message = 'Room edited successfully'
            room_members_str = ",".join(new_members)
        return render_template('edit_room.html', room=room, room_members_str=room_members_str, message=message)
    else:
        return "Room not found", 404


@app.route('/rooms/<room_id>/')
@login_required
def view_room(room_id):
    room = get_room(room_id)
    if room and is_room_member(room_id, current_user.username):
        room_members = get_room_members(room_id)
        messages = get_messages(room_id)
        usermode = get_mode_for_user(current_user.username)
        return render_template('view_room.html', username=current_user.username, room=room, room_members=room_members,
                               messages=messages, usermode=usermode)
    else:
        return "Room not found", 404


@app.route('/rooms/<room_id>/messages/')
@login_required
def get_older_messages(room_id):
    room = get_room(room_id)
    if room and is_room_member(room_id, current_user.username):
        page = int(request.args.get('page', 0))
        messages = get_messages(room_id, page)
        return dumps(messages)
    else:
        return "Room not found", 404


@socketio.on('send_message')
def handle_send_message_event(data):
    # app.logger.info("{} has sent message to the room {}: {}".format(data['username'],
    #                                                                 data['room'],
    #                                                                 data['message']))
    data['created_at'] = datetime.now().strftime("%d %b, %H:%M")
    
    indices = sentences_to_indices(np.array([data['message']]), word_to_index, 10)
    data['sentiment'] = str(np.argmax(model.predict(indices)))
    if(data['sentiment']=='0'):
        data['sentiment']=128150
    elif(data['sentiment']=='1'):
        data['sentiment']=9917
    elif(data['sentiment']=='2'):
        data['sentiment']=128512
    elif(data['sentiment']=='3'):
        data['sentiment']=128529
    elif(data['sentiment']=='4'):
        data['sentiment']=127860
    else:
        pass
    save_message(data['room'], data['message'], data['username'], data['sentiment'])

    socketio.emit('receive_message', data, room=data['room'])


@socketio.on('join_room')
def handle_join_room_event(data):
    app.logger.info("{} has joined the room {}".format(data['username'], data['room']))
    join_room(data['room'])
    socketio.emit('join_room_announcement', data, room=data['room'])


@socketio.on('leave_room')
def handle_leave_room_event(data):
    app.logger.info("{} has left the room {}".format(data['username'], data['room']))
    leave_room(data['room'])
    socketio.emit('leave_room_announcement', data, room=data['room'])


@login_manager.user_loader
def load_user(username):
    return get_user(username)


if __name__ == '__main__':
    socketio.run(app, host='127.0.0.1', port=4040, debug=True)
 
