#
# NAME
#   CreateExamplePage.py
#
# SYNOPSIS
#   This script converts an ipython notebook into a PHP webpage for the MUQ website
#
# DESCRIPTION
#   This script takes two input arguments: the ipynb input file and the PHP output file.
#   Calling this script will involve a command like:
#
#   CreateExamplePage.py -i <inputfile> -o <outputfile>
#

import os
import re
import json
from nbconvert.filters.markdown import markdown2html
import html

import sys, getopt

def PrintUsage():
    print('  CreateExamplePage.py -i <inputfile> -o <outputfile>\n')
    sys.exit(2)


def ProcessNotebook(inputfile):

    # create a string to hold the html that we can include in the MUQ website
    htmlStr = ''
    allCode = ''

    # loop through all the cells in the ipynb file
    with open(inputfile,'r') as f:
        contents = json.load(f)
    for cell in contents['cells']:

        # CODE CELLS
        if(cell['cell_type']=='code'):
            cellLines = cell['source']

            # check to make sure this cell isn't empty
            if(len(cellLines)>0):

                # look for magic lines at the beginning of the cell
                skipOut = False
                skipCode = False
                if(cellLines[0][0]=='%'):

                    # if this cell is writing to a file, we will skip the output
                    if(len(cellLines[0])>10):
                        if(cellLines[0][0:11]=='%%writefile'):
                            skipOut = True

                    if(len(cellLines[0])>5):
                        if(cellLines[0][0:6]=='%%bash'):
                            skipCode = True

                    # remove the magic line
                    cellLines = cellLines[1::]


                htmlStr += '<pre class="prettyprint">\n'
                for line in cellLines:
                    safeStr = html.escape(line)
                    if(not skipCode):
                        allCode += safeStr
                    htmlStr += safeStr
                htmlStr += '\n</pre>\n\n'

                if(not skipCode):
                    allCode += '\n\n'

                if(not skipOut):
                    for outCell in cell['outputs']:
                        if 'text' in outCell:
                            outLines = outCell['text']
                            if(len(outLines)>0):
                                htmlStr += '<pre class="prettyprint lang-bash" style="background-color:#D0D0D0">\n'
                                for line in outLines:
                                    htmlStr += html.escape(line)
                                htmlStr += '\n</pre>\n\n'
                        elif 'data' in outCell:
                            if('image/png' in outCell['data']):
                                htmlStr += '<img src="data:image/png;base64,'
                                htmlStr += outCell['data']['image/png']
                                htmlStr += '"></img>\n\n'

        # MARKDOWN CELLS
        elif(cell['cell_type']=='markdown'):
            cellLines = cell['source']

            # check to make sure this cell isn't empty
            if(len(cellLines)>0):
                allLines = ''
                for line in cellLines:
                    allLines += line
                htmlStr += re.sub('<a class="anchor-link" href="[^<]+</a>','',markdown2html(allLines))

        else:
            raise ValueError("ERROR: Invalid cell type, we can currently only handle 'code' and 'markdown'.")

    # add the entire code to the html string
    htmlStr += '<h2>Completed code:</h2>'
    htmlStr += '<pre class="prettyprint" style="height:auto;max-height:400px;">\n'
    htmlStr += allCode
    htmlStr += '\n</pre>\n\n'

    return htmlStr


if __name__=='__main__':

    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        PrintError()
        PrintUsage()

    for opt, arg in opts:
        if opt == '-h':
            PrintUsage()
        elif opt in ("-i", "--ifile"):
            inputfile = os.path.abspath(arg)
        elif opt in ("-o", "--ofile"):
            outputfile = os.path.abspath(arg)

    if(len(opts)!=2):
        print('\nInvalid number of options.  Correct usage is:')
        PrintUsage()

    htmlStr = ProcessNotebook(inputfile)

    # create any necessary directories, open up a file, and write the webpage
    fullDir = '/'.join(outputfile.split('/')[0:-1])

    if(not os.path.exists(fullDir)):
        os.makedirs(fullDir)

    with open(outputfile,'w') as fout:
        fout.write(htmlStr)
