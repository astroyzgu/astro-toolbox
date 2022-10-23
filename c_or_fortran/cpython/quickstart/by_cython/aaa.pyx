cdef int an[10]
cdef int n = 123
cdef int *pn = &n
print("%d \n"% pn[0])
