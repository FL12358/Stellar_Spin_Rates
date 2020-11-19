import requests

def Notificate(text):
    slack_token = "xoxb-1032613313316-1023077206497-uXkkO5RnhfyRDOMYb0ikwfzw"
    
    data = {
        'token': slack_token,
        'channel': 'U010Z1TLZL5',
        'as_user': True,
        'text': text
    }
    
    requests.post(url='https://slack.com/api/chat.postMessage',
                  data=data)