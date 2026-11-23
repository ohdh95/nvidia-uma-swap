#define _GNU_SOURCE
#include <dlfcn.h>
#include <sys/ioctl.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
int ioctl(int fd, unsigned long request, ...) {
 static int (*real_ioctl)(int,unsigned long,...);
 if(!real_ioctl) real_ioctl=dlsym(RTLD_NEXT,"ioctl");
 va_list ap;va_start(ap,request);void* arg=va_arg(ap,void*);va_end(ap);
 int observe=arg && _IOC_TYPE(request)==0x46 && _IOC_NR(request)==0x2b && (_IOC_SIZE(request)==32 || _IOC_SIZE(request)==48);
 uint32_t class_id=0;
 if(observe)memcpy(&class_id,(char*)arg+12,4);
 int result=real_ioctl(fd,request,arg);int saved_errno=errno;
 if(observe){
  uint32_t status=0;memcpy(&status,(char*)arg+(_IOC_SIZE(request)==48 ? 40 : 28),4);
  char buf[160];int n=snprintf(buf,sizeof(buf),"OBSERVED RM_ALLOC class=0x%08x status=0x%x syscall_result=%d\n",class_id,status,result);
  write(2,buf,n);
 }
 errno=saved_errno;return result;
}
