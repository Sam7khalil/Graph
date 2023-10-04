from CONWAY_data import conway
#print (len(conway))
#print (conway[0])


def selfname(str):
    return "".join(c for c in str if c in "0123456789")


def numeric(str):
    return all(c in "0123456789" for c in str)


num_of = dict((name, num) for fr, num, name, val in conway)
name_of = dict((num, name) for fr, num, name, val in conway)

out_nbrs = dict((i, []) for i in range(1, 92+1))
in_nbrs = dict((i, []) for i in range(1, 92+1))
for fr, num, name, val in conway:
    nbrs = val.split('.')
#    print nbrs,num
    for n in nbrs:
        if not numeric(n):
            nn = num_of[n]
            out_nbrs[num+1].append(nn)
            in_nbrs[nn].append(num+1)


def indexname(nam):
    return "%s(%d)" % (nam, num_of[nam])


def indexnum(n):
    return "%s(%d)" % (name_of[n], n)


def indexprint(n):
    if numeric(n):
        return n
    else:
        return indexname(n)


for fr, num, name, val in conway:
    if len(selfname(val)) % 2:
        print('- ', end="")
    else:
        print('+ ', end="")
    print("%2d" % num, "%-2s" % name, "%11.5f" % fr, selfname(val), end='')
    if in_nbrs[num]:
        print(" ["+",".join(indexnum(n) for n in in_nbrs[num]), end=']')
    nbrs = val.split('.')
    outval = (".".join(indexprint(n) for n in nbrs))
    if outval != val:
        print(" ::", outval, end="")
    print()


cnew = [(selfname(val), num, name)
        for fr, num, name, val in conway]
#for x in sorted(cnew):    print(x)

for fr, num, name, val in conway:
    folge = "[" + ",".join(c for c in val if c in "0123456789")+"]"
    print("> cName", folge, "=(", num, ', "'+name+'")')
