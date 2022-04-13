import os
import sys
import PySimpleGUI as sg

dir_root = os.path.dirname(os.path.dirname(__file__))
if dir_root not in sys.path:
    sys.path.append(dir_root)

from imgp_agent.face_shapekey_comparetor import FaceShapekeyComparetor

sg.theme('Light Blue 2')

layout = [[sg.Text('选择图片')],
          [sg.Button('OK'), sg.Button('Cancel')],
          [sg.Text('File 1', size=(5, 1)), sg.Input(size=(40, 1)), sg.FileBrowse(key='-File1-')],
          [sg.Text('File 2', size=(5, 1)), sg.Input(size=(40, 1)), sg.FileBrowse(key='-File2-')],
          [sg.Text('File 3', size=(5, 1)), sg.Input(size=(40, 1)), sg.FileBrowse(key='-File3-')]
        ]

window = sg.Window('File Compare', layout)

event, values = window.read()

while True:
    event, values = window.read()
    print(f'Event: {event}')
    
    if event == 'OK':
        fileNames = []
        if len(values[0]) > 0:
            fileNames.append(values[0])
        if len(values[1]) > 0:
            fileNames.append(values[2])
        if len(values[2]) > 0:
            fileNames.append(values[2])
        print('select file: ', fileNames)
        fc = FaceShapekeyComparetor()
        fc.process(fileNames)
        
    elif event in (None, 'Cancel'):
        # User closed the Window or hit the Cancel button
        break
    print(str(values))
    
window.close()
print(f'You clicked {event}')
