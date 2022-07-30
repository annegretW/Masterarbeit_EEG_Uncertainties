import html

def ProcessScript(inputfile):

    # create a string to hold the html that we can include in the MUQ website
    htmlStr = ''
    allCode = ''

    # loop through all the cells in the ipynb file
    with open(inputfile,'r') as f:
        for line in f.readlines():
            safeStr = html.escape(line)
            allCode += safeStr

    htmlStr += '<pre class="prettyprint" style="height:auto;max-height:400px;">\n'
    htmlStr += allCode
    htmlStr += '\n</pre>\n\n'

    return htmlStr
