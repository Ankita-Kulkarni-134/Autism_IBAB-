'''''
 5.5. Make a dictionary called plain with the key-value pairs 'a': 1, 'b': 2, and 'c':
 3, and then print it.

plain = { 'a': 1, 'b': 2,'c': 3}
print(plain)
'''''
'''''
5.6. Make an OrderedDict called fancy from the same pairs listed in 5.5 and print it.
 Did it print in the same order as plain?

from collections import OrderedDict
fancy = OrderedDict({ 'a': 1, 'b': 2,'c': 3})
print(fancy)

#yes it is printing it in the same way 
'''''

'''''
 5.7. Make a defaultdict called dict_of_lists and pass it the argument list. Make
 the list dict_of_lists['a'] and append the value 'something for a' to it in one
 assignment. Print dict_of_lists['a'].

from collections import defaultdict
dict_of_lists = defaultdict(list)
dict_of_lists['a'].append('Ankita ')
print(dict_of_lists['a'])
'''''