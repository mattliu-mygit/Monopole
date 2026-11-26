"""Monopole as a Prompting Time plugin: the JSON actions `prompting-time-plugin.json` declares.

Prompting Time runs each action as a short-lived command and reads what it prints. The actions
read the same run database the server uses, keep their own files in the plugin's state folder
(`$XDG_STATE_HOME/monopole`, which Prompting Time watches), and never call CES or a model.
"""
