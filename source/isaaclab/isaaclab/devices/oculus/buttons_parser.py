def parse_buttons(text):
    split_text = text.split(',')
    buttons = {}
    if 'R' in split_text: # right hand if available
        split_text.remove('R') # remove marker
        buttons.update({'A': False,
                        'B': False,
                        'RThU': False, # indicates that right thumb is up from the rest position
                        'RJ': False, # joystick pressed
                        'RG': False, # boolean value for trigger on the grip (delivered by SDK)
                        'RTr': False # boolean value for trigger on the index finger (delivered by SDK)
                        })
        # besides following keys are provided:
        # 'rightJS' / 'leftJS' - (x, y) position of joystick. x, y both in range (-1.0, 1.0)
        # 'rightGrip' / 'leftGrip' - float value for trigger on the grip in range (0.0, 1.0)
        # 'rightTrig' / 'leftTrig' - float value for trigger on the index finger in range (0.0, 1.0)

    if 'L' in split_text: # left hand accordingly
        split_text.remove('L') # remove marker
        buttons.update({'X': False, 'Y': False, 'LThU': False, 'LJ': False, 'LG': False, 'LTr': False})
    for key in buttons.keys():
        if key in list(split_text):
            buttons[key] = True
            split_text.remove(key)
    for elem in split_text:
        split_elem = elem.split(' ')
        if len(split_elem) < 2:
            continue
        key = split_elem[0]
        value = tuple([float(x) for x in split_elem[1:]])
        buttons[key] = value
    return buttons
    
    