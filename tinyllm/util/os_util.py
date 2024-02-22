import psutil
import getpass
import os
import pickle
import platform
import shutil
import stat
import sys
import zipfile
from pathlib import Path
from typing import Union

import json


def loadJson(json_path=None):
    with open(json_path) as json_file:
        dic = json.load(json_file)
        return dic


def saveJson(dic=None, save_to_path=None):
    ensureDir(save_to_path)
    with open(save_to_path, 'w') as f:
        json.dump(dic, f)


def ensureDir(dir_path):
    if isFilePath(dir_path):
        dir_path = getParentDir(dir_path)
    if isDirPath(dir_path) and not dirExists(dir_path):
        os.makedirs(dir_path)


def joinPaths(paths: list):
    # Remove '/' at the beginning of all paths
    paths = [path[1:] if path.startswith('/') and i != 0 else path for i, path in enumerate(paths)]
    return os.path.join(*paths)


def getBaseName(path):
    return os.path.basename(path)


def getCurrentDirPath():
    return os.getcwd()


def getUserHomePath():
    return str(Path.home())


def getUsername():
    return getpass.getuser()


def getPythonExecutablePath():
    return sys.executable


def getOS():
    return platform.system()


def dirExists(path):
    return os.path.isdir(path)


def fileExists(path):
    return os.path.isfile(path)


def isFilePath(path):
    split_path = str(path).split('/')
    return '.' in split_path[len(split_path) - 1]


def isDirPath(path):
    split_path = str(path).split('/')
    return '.' not in split_path[len(split_path) - 1]


def getPythonVersion():
    return sys.version[:5]


def pathExists(path):
    return sum([fileExists(path), dirExists(path)]) > 0


def on_rm_error(func, path, exc_info):
    # sizer_path contains the sizer_path of the sizer_path that couldn't be removed
    # let's just assume that it's read-only and unlink it.
    os.chmod(path, stat.S_IWRITE)
    os.unlink(path)


def remove(path):
    if fileExists(path):
        os.remove(path)
    elif dirExists(path):
        shutil.rmtree(path, onerror=on_rm_error)



def runCommand(command):
    result = os.system(command)
    if result != 0:
        raise Exception(f'Command failed : "{command}"')
    else:
        return result


def writeFile(path, lines):
    with open(path, 'w') as fOut:
        for line in lines:
            fOut.write(line)
            fOut.write("\n")


def copyFromTo(from_path, to_path, overwrite=True, ignore_files: list = None):
    if fileExists(from_path):
        if overwrite:
            remove(to_path)
            ensureDir(getParentDir(to_path))
            shutil.copyfile(from_path, to_path)
    elif dirExists(from_path):
        if overwrite:
            remove(to_path)
        shutil.copytree(from_path, to_path)


def getParentDir(path):
    return os.path.dirname(path)


def listDir(path,
            formats='', recursive=False):
    if isinstance(formats, str):
        formats = [formats]
    if recursive is True:
        listOfFile = os.listdir(path)
        allFiles = list()
        for entry in listOfFile:
            fullPath = os.path.join(path, entry)
            if os.path.isdir(fullPath):
                allFiles = allFiles + listDir(fullPath, recursive=recursive, formats=formats)
            else:
                for format in formats:
                    if fullPath.endswith(format):
                        allFiles.append(fullPath)
                        break
        return allFiles
    else:
        return [os.path.join(path, i) for i in os.listdir(path) if any(i.endswith(format) for format in formats)]

def walkDir(paths: Union[str, list], extension="", ignore=[]) -> list:
    if isinstance(paths, str):
        paths = [paths]
    files = []
    for dir_path in paths:
        for current_dir_path, current_subdirs, current_files in os.walk(dir_path):
            for aFile in current_files:
                if aFile.endswith(extension):
                    txt_file_path = str(os.path.join(current_dir_path, aFile))
                    if not any(word in txt_file_path for word in ignore):
                        files.append(txt_file_path)
    return list(files)


def copyDir(src, dst):
    src = Path(src)
    dst = Path(dst)
    ensureDir(dst)
    for item in os.listdir(src):
        s = src / item
        d = dst / item
        if dirExists(s):
            copyDir(s, d)
        else:
            shutil.copy2(str(s), str(d))


def zipFiles(src: list, dst: str, arcname=None):
    zip_ = zipfile.ZipFile(dst, 'w')
    for i in range(len(src)):
        if arcname is None:
            zip_.write(src[i], os.path.basename(src[i]), compress_type=zipfile.ZIP_DEFLATED)
        else:
            zip_.write(src[i], arcname[i], compress_type=zipfile.ZIP_DEFLATED)
    zip_.close()


def zipDir(path, save_to):
    zip_ = zipfile.ZipFile(save_to, 'w')
    for root, dirs, files in os.walk(path):
        for file in files:
            zip_.write(os.path.join(root, file),
                       os.path.relpath(os.path.join(root, file),
                                       os.path.join(path, '../..')))
    zip_.close()


def recursiveOverwrite(src, dest, ignore=None):
    if dirExists(src):
        if not dirExists(dest):
            ensureDir(dest)
        files = listDir(src)
        if ignore is not None:
            ignored = ignore(src, files)
        else:
            ignored = set()
        for f in files:
            if f not in ignored:
                recursiveOverwrite(joinPaths([src, f]),
                                   joinPaths([dest, f]),
                                   ignore)
    else:
        shutil.copyfile(src, dest)



def loadPickle(file_path):
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)


def savePickle(object, file_path):
    with open(file_path, 'openpyxl_sizer_workbook') as handle:
        pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)


def getComputerStats():
    gb_divider = (1024.0 ** 3)
    stats = {
        "cpu_percent": psutil.cpu_percent(),
        "ram_available": psutil.virtual_memory().available/gb_divider,
        "ram_total": psutil.virtual_memory().total/gb_divider,
        "ram_percent": psutil.virtual_memory().available * 100 / psutil.virtual_memory().total
    }
    return stats


def getTempDir(folder_name, ensure_dir=True):
    path = joinPaths([getUserHomePath(),"tmp",folder_name])
    if ensure_dir: ensureDir(path)
    return path
