import owncloud


def main():
    oc = owncloud.Client("http://localhost:8080/")
    oc.login('admin', 'admin')

    files = oc.list('testdir/', 'infinity')
    for f in files:
        print(f.get_name())
        print(oc.get_file_contents(f.get_path()))

    oc.delete('testdir/')


if __name__ == '__main__':
    main()
