css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 40%;
  min-width: 80px;
}
.chat-message .avatar img {
  width: 120px;
  height: 120px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://media.istockphoto.com/id/1409839764/vector/cute-little-robot-smiling-robotics-and-technology-kawaii-robot.jpg?s=612x612&w=0&k=20&c=qV3NO5VjN6UWdqWXDaKoFEAxt2o0ak0_jQRmM_JVGV4=">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://imgcdn.stablediffusionweb.com/2025/8/17/85c6454a-e1fd-41ed-918e-7a3fb37eb167.jpg">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''