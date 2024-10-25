#include <stdio.h>

#include <pthread.h>

#include <semaphore.h>

#include <unistd.h>

sem_t rNV;
sem_t rNG;
sem_t site;

int nV=0;
int nG=0;

void* Guide(void* arg){

long tid;
tid = (long)arg;

	sem_wait(&rNG);
	nG++;
	printf("\nGuide:%ld Entered...\n", tid);
	sem_post(&rNG);
	sem_post(&site);
	sem_post(&site);
	sem_post(&site);
	int i=0;
	while(i==0){
	sleep(420);
	sem_wait(&rNG);
	sem_wait(&rNV);
	if((nG-1)*3>=nV){
		printf("\nGuide:%ld Exiting...\n", tid);
		nG--;
		i=1;}
	sem_post(&rNG);
	sem_post(&rNV);

	}
}

void* Visitor(void* arg){

long tid;
tid = (long)arg;
	sem_wait(&site);
	sem_wait(&rNV);
	nV++;
	printf("\nVisitor:%ld Entered...\n", tid);
	sem_post(&rNV);
	sleep(300);
	sem_wait(&rNV);
	nV--;
	printf("\nVisitor:%ld Exiting...\n", tid);
	sem_post(&site);
	sem_post(&rNV);
}
void* Inspector(void* arg){
	sem_wait(&rNV);
	sem_wait(&rNG);
	if(nG*3>=nV){
		printf("\nInspector verified the ratio\n");}
	else printf("\nSomething goes wrong\n");
	sem_post(&rNG);
	sem_post(&rNV);
}

int main(){

sem_init(&site,0,0);
sem_init(&rNV,0,1);
sem_init(&rNG,0,1);
pthread_t v[30],g[3],i1,i2;
pthread_create(&g[0],NULL,Guide, (void *)1);
sleep(60);
pthread_create(&v[0],NULL,Visitor, (void *)1);
sleep(60);
pthread_create(&v[1],NULL,Visitor, (void *)2);
sleep(60);
pthread_create(&v[2],NULL,Visitor, (void *)3);
sleep(60);
pthread_create(&v[3],NULL,Visitor, (void *)4);
sleep(60);
pthread_create(&g[1],NULL,Guide, (void *)2);
pthread_create(&v[4],NULL,Visitor, (void *)5);
sleep(60);
pthread_create(&v[5],NULL,Visitor, (void *)6);
sleep(60);
pthread_create(&v[6],NULL,Visitor, (void *)7);
sleep(60);
pthread_create(&v[7],NULL,Visitor, (void *)8);
sleep(60);
pthread_create(&i1,NULL,Inspector,NULL);
pthread_create(&v[8],NULL,Visitor, (void *)9);
sleep(60);
pthread_create(&g[2],NULL,Guide, (void *)3);
pthread_create(&v[9],NULL,Visitor, (void *)10);
sleep(60);

for (int j=10; j<24; j++) {
pthread_create(&v[j],NULL,Visitor, (void *)(int)(j+1));
sleep(60);
}
pthread_create(&i2,NULL,Inspector,NULL);
for (int j=24; j<30; j++) {
pthread_create(&v[j],NULL,Visitor, (void *)(int)(j+1));
sleep(60);
}

pthread_join(g[0],NULL);
pthread_join(v[0],NULL);
pthread_join(v[1],NULL);
pthread_join(v[2],NULL);
pthread_join(v[3],NULL);
pthread_join(g[1],NULL);
pthread_join(v[4],NULL);
pthread_join(v[5],NULL);
pthread_join(v[6],NULL);
pthread_join(v[7],NULL);
pthread_join(i1,NULL);
pthread_join(v[8],NULL);
pthread_join(g[2],NULL);
pthread_join(v[9],NULL);
for (int k=10; k<24; k++) {
pthread_join(v[k],NULL);
}
pthread_join(i2,NULL);
for (int k=24; k<30; k++) {
pthread_join(v[k],NULL);
}
sem_destroy(&site);
sem_destroy(&rNV);
sem_destroy(&rNG);
return 0;
}
