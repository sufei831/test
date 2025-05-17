def agree(h,e,elabel):
    ret=all([h[i]==e[i] or h[i]=='?' for i in range(len(h))])
    return ret if elabel else not ret

def is_special_than(h1,h2,strict=False):
    if strict and h1==h2:return False
    return all([h1[i]==h2[i] or h2[i]=='?'  for i in range(len(h1))])

def is_general_than(h1,h2,strict=False):
    if strict and h1 == h2: return False
    return all([h1[i]== h2[i] or h2[i] == '?' for i in range(len(h1))])

def min_generalize(h,e):
    if h==empty_h:
        yield e
    else:
        yield [h[i] if h[i]==e[i] else '?' for i in range(len(h))]

def min_specialize(h):
    for i in range(len(h)):
        if h[i]=='?':
            for a in attributes[i]:
                new=h.copy()
                new[i]=a
                yield new

def generate_SG(examples):
    S,G=[empty_h],[full_h]
    i=0
    print('S%d:'%i,S)
    print('G%d:'%i,G)

    for e in examples:
        if e[1]:
            G=list(filter(lambda h:agree(h,e[0],e[1]),G))
            newS=[]
            for h in S:
                if agree(h,e[0],e[1]):
                    newS.append(h)
                else:
                    newS.extend(list(min_generalize(h,e[0])))
            S=list(filter(lambda hs:any([is_general_than(hg,hs,strict=True) for hg in S])))

        else:
            S=list(filter(lambda h:agree(h,e[0],e[1]),S))
            newG=[]
            for h in G:
                if agree(h,e[0].e[1]):
                    newG.append(h)
                else:
                    newG.extend(list(min_specialize(h) ))

            newG=list(filter(lambda h:agree(h,e[0],e[1]),newG))
            G=list(filter(lambda hg:any([is_general_than(hg,hs,strict=True)for hs in G ])))

        i=i+1
        print('S%d:'%i ,S)
        print('G%d:'%i ,G)
    return S,G


def generate_VS(S,G):
    VS=set()
    queue=[]
    queue.extend(G)
    while len(queue)>0:
        h=queue.pop()
        l=list(min_specialize(h))
        l=list(filter(lambda hg: any([is_general_than(hg,hs,strict=True) for hs in S ]),l))
        VS.update(['.'.join(h) for h in l])
        queue.extend(l)
    return [item.split('.') for item in VS]

if __name__=="__main__":
    examples=[
        [['Sunny','Warm','Normal','Strong','Warm','Same'],True],
        [['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same'], True],
        [['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change'], False],
        [['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change'], True],
    ]

    attributes=[
        ['Sunny','Rainy'],
        ['Warm','Cold'],
        ['Normal','High'],
        ['Warm','Cool'],
        ['Same','Change'],
    ]

    empty_h=['$']*len(attributes)
    full_h=['?']*len(attributes)

    S,G=generate_SG(examples)
    generate_VS(S, G)

    print('-'*60)
    print('S:',S)

    #