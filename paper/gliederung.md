## was ist federated learning

basic stuff

## welche arten gibt es

- cross device
- cross silo


## arten von federated strategien

server step (clients senden gradients, server aggregiert gradients und macht einen optimizer step) z.B. FedSDG

client step (client macht step und sendet neue weights an server, server aggregiert weights zu neuem model) z.B. FedAvg

server and client steps (client macht steps mit lokaler lr, aggregiert lokale gradienten, sendet gradienten an server, server aggregiert empfangene client gradienten und macht dann einen step) z.B. FedExp oder FedHyper?

## arten von datasets

- IID
- NonIID


## arbeiten in die richtung 
- hier auf fedhyper wegen learning rate scheduling eingehen (klassich oft learning rate scheduler, fed avg jedoch nicht)
- das paper mit IID vs NonIID weight divergence

## eigene untersuchungen
basis model - mit spec augmentation von nemo
datenaufbereitung - entfernung von , und so

cross device, ...
finetuning von asr modellen., 

## erkenntisse hyperparamter
viele clients pro round aber wenig round schlecht, quasi wie ein mega aggregierter batch ueber das gesamte datenset im klassichen
bild mit strichen einfeugen wo parallele striche ein aggregierter batch sind. soll zeigen das im klassichen auch aggregiert wird, nur weitaus weniger als full clent fededrated learning.
ideal waere quasi ein client pro round um ans klassiche ranzukommen

zeigen mathematisch



## erkenntise typ der daten IID/NonIID

timit dataset wohl nicht wirkklich NonIID da keien wirkliche unterschiede in performance festgestellt wurde. 

weight divergence vlt doch anders? mal gucken


## erkenntisse data augmentation

bisher nicht gemacht



## other cheks that can be done
in dieser arbeit wurde sich an dem vorher finegetunten model orientiert. hier kann weiter analysieren wie data agumentaitn federated learning beeinflusst: 
removing spec augmentation

tts data generation
