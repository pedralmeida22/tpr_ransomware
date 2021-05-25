import os
import shutil
import pyinputplus as pyip

from encriptor import Encryptor

# path para pasta sincronizada
PATH = "/home/pedralmeida/Documents/owncloud/"


def dir_txt_files(dir_name):
    """ Creates a list of all files in the folder 'dir_name'"""

    fileList = []
    '''creates an empty list'''
    for file in os.listdir(dir_name):
        '''for all files in the directory given'''
        dirfile = os.path.join(dir_name, file)
        '''creates a full file name including path for each file in the directory'''
        if os.path.isfile(dirfile) and os.path.splitext(dirfile)[1][1:] == 'txt':
            '''if the full file name above is a file and it ends in 'txt' it will be added to the list 
            created above '''
            fileList.append(dirfile)
    return fileList


def put_files(directory='testdir'):
    os.chdir("/home/pedralmeida/Documents/tpr_ransomware")    # dir com ficheiros a copiar
    for f in dir_txt_files(os.getcwd()):
        # print(f)
        shutil.copy(f, PATH + directory)
    print("ficheiros copiados")


def check_dir(path, directory="testdir"):
    p = os.path.join(path, directory)
    if not os.path.isdir(p):
        print("nao existe, cria e poe ficheiros")
        os.mkdir(p)
        put_files(directory)
    else:
        print("ja existe")


def main():

    while True:
        op = pyip.inputMenu(['New dir', 'Cipher', 'Delete', 'Exit'], "Option->\n", numbered=True, limit=3)

        if op == 'New dir':
            os.chdir(PATH)
            new_dirname = pyip.inputStr("Dir name?\n", limit=20, blank=True)
            if new_dirname == '':
                check_dir(os.getcwd())
            else:
                check_dir(os.getcwd(), new_dirname)  # segundo argumento -> nome para criar um novo diretorio
            print("Dir '" + dirname + "' created")

        elif op == "Delete":
            os.chdir(PATH)
            dirname = pyip.inputStr("Dir name?\n", limit=20)
            shutil.rmtree(PATH + dirname)
            print("Dir '" + dirname + "' deleted")

        elif op == 'Cipher':
            os.chdir("/home/pedralmeida/Documents/tpr_ransomware")

            encryptor = Encryptor()
            mykey = encryptor.key_create()
            encryptor.key_write(mykey, 'mykey.key')
            loaded_key = encryptor.key_load('mykey.key')

            dirname = pyip.inputStr("Dir name?\n", limit=20)
            if os.path.exists(PATH + dirname):
                os.chdir(PATH + dirname)
            else:
                print("Path: '" + PATH + dirname + "' nao existe")
                continue

            for f in dir_txt_files(os.getcwd()):
                print(f)
                encryptor.file_encrypt(loaded_key, f, f)

        elif op == 'Exit':
            break


if __name__ == '__main__':
    main()
