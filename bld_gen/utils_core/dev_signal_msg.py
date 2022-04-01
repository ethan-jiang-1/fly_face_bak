_initialized = False

if not _initialized:
    try:
        import blinker
        _SC_SIG_MSG = blinker.signal('sc_sig_msg')
    except Exception as ex:
        _SC_SIG_MSG = None 
        print("Exception occured", ex)
    _initialized = True

def get_sig_msg():
    return _SC_SIG_MSG

def _sc_send_dev_msg(msg):
    sig_msg = get_sig_msg()
    if sig_msg is None:
        return None
    ret = sig_msg.send('sender', message=msg)
    return ret

def _handle_msg(sender, **kwargs):
    print("get_msg", sender, kwargs)

def _test_register_msg_receiver():
    sig_msg = get_sig_msg()
    if sig_msg is None:
        return None

    print("*register message reciver", sig_msg)
    sig_msg.connect(_handle_msg)

    _sc_send_dev_msg("init_msg")

def _test_send_dev_msgs():
    sig_msg = get_sig_msg()
    if sig_msg is None:
        return None

    print("*send batch of messages", sig_msg)
    for i in range(10):
        _sc_send_dev_msg("hello {}".format(i))


if __name__ == "__main__":
    _test_register_msg_receiver()
    _test_send_dev_msgs()
