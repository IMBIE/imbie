#!/usr/bin/python3
import os, json, sys

def iom_info(data, name):
	opts = [
		'thick-and-vel'
	]
	for opt in opts:
		info = data.get(opt)
		if info is None: continue
		for sheet in info:
			print(name,'iom',opt,sheet,sep=',')

def ra_info(data, name):
	opts = [
		'volume-change',
		'firn-volume-change'
	]
	for opt in opts:
		info = data.get(opt)
		if info is None: continue
		for sheet in info:
			print(name,'ra',opt,sheet,sep=',')

def gmb_info(data, name):
	opts = [
		'mass-change-sheets',
		'no-gia-sheets',
		'no-degree-sheets',
		'no-c20-sheets',
		'cm-water-sheets'
	]
	for opt in opts:
		info = data.get(opt)
		if info is None: continue
		for sheet in info:
			print(name,'gmb',opt,sheet,sep=',')

def smb_info(data, name):
	opts = [
		'component-fields-sheets',
		'gridded-component-fields-sheets'
	]
	for opt in opts:
		info = data.get(opt)
		if info is None: continue
		for sheet in info:
			print(name,'smb',opt,sheet,sep=',')

def get_exps(fpath):
	funcs = {
		'IOM': iom_info,
		'RA': ra_info,
		'GMB': gmb_info,
		'SMB': smb_info
	}

	with open(fpath) as f:
		data = json.load(f)
		exps = data.get('additional')
		if exps is None:
			return
		grp = data['group']
		name = data['username']

		if grp in funcs:
			f = funcs[grp]
			f(exps, name)

def search(root=None):
	if root is None:
		root = os.getcwd()

	for path, _, files in os.walk(root):
		if '.answers.json' not in files:
			continue
		fpath = os.path.join(path, '.answers.json')
		get_exps(fpath)

if __name__ == "__main__":
	search(sys.argv[1])