import owncloud


def main():
    oc = owncloud.Client("http://localhost:8080/")
    oc.login('admin', 'admin')
    user = oc.get_users()
    print('user: ', user)

    dir_name = 'testdir'

    oc.mkdir(dir_name)
    oc.put_file(dir_name + '/', 'proj/something.txt')
    oc.put_file(dir_name + '/', 'proj/note.txt')

    files = oc.list('.', 'infinity')
    for f in files:
        print(f.get_name())


if __name__ == '__main__':
    main()
