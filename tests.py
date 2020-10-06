from storage import Storage

def test_add():
    st=Storage()
    st.add('a', 1)
    assert st.get('a')!=1, "Value for the key {} does not exist"

def test_remove():
    db = Storage({'a': 1, 'b': 2})
    key = 'a'
    db.remove(key)
    val = st.get(key)
    assert val is None, "The <key,value> pair hasn't been removed"
    key = 'с'
    try:
        db.remove(key)
    except KeyError:
        pass
    else:
        raise Exception

def test_set():
    st = Storage({'a': 1, 'b': 2})
    key = 'a'
    val = 3
    st.set(key,val)
    assert st.data[key] == 3, "Value for the key {} is not equal to set {} ".format(key, val)

def test_get():
    db = Storage({'a': 1, 'b': 2})
    key = 'b'
    val = db.get(key)
    assert val == 2, "Value for the key {} is not equal to expected".format(key)
    key = 'c'
    val = db.get(key)
    assert val is None, "Value for an unexisting key is not None"

def run_tests():
    test_add()
    test_remove()
    test_set()
    test_get()

if __name__ == "__main__":
    run_tests()
