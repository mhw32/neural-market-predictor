""" The purpose of the file is to read in a .mat file from 
	matlab's web scrapper and load it into a postgres DB """

import scipy.io, psycopg2 
import sys, os

def mat2db():
	if len(sys.argv) != 1:
		return
	# read in the matfile
	matfile = sys.argv[0]
	newdata = scipy.io.load(matfile)

	try:
		psycopg2.connect("dbname='%s' user='%s' password='%s'", os.environ['FINANCE_DB_NAME'], \
			os.environ['FINANCE_DB_USER'], os.environ['FINANCE_DB_PWD'])
	except:
		print 'I am unable to connect to the database'

	cur = conn.cursor()
	# TO-DO: Now I'm ready to insert data.

if __name__ == '__main__':
	mat2db()
