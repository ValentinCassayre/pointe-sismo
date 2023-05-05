from picker import *
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

# Chargement des données
starttime = UTCDateTime(2021, 6, 26, 2, 58, 0)
duration = 5*60  # secondes
station = "ILLF" # station d'Illkirch
client = Client("RESIF")  # Réseau français
S = client.get_waveforms(network="FR", station="ILLF", location="00", channel="HH*", starttime=starttime, endtime=starttime+duration)
inventory = client.get_stations(network="FR", station=station, channel="HH*", starttime=starttime, level="response")

# Correction du signal
pre_process_data(S, inventory)

# On garde les arrays des temps et des amplitudes
T = S.select(channel="HHZ")[0]
times = T.times(type='relative')
data = T.data

# Paramètrage de la taille de la fenêtre
tlta = 20
tsta = 1
nsta, nlta = nstalta_from_times(times, tsta, tlta)

# Calcul des enveloppes
sqa_data = data**2
fc_stewart = mdx_stewart(times, data, 10)
allen_data = allen_envelope_np(data)
baer_data = baer_envelope_np(times, data)

# Calcul des fonctions caractéristiques
fc_allen = stalta_allen_np(allen_data, nsta, nlta)
fc_allen_recursiv = stalta_recursiv(allen_data, nsta, nlta)
z = z_swindell_snell(sqa_data, nsta)
fc_baer = z_baer_kradolfer(baer_data, nsta)

# Détection
threshold = 5
potentials = potential_waves(times, z, threshold=threshold, tol=1, delta=3)
earthquake = any(potentials)

# Pointage et affichage
if earthquake:
    earthquake_loc = potential_waves(times, z, threshold, tol=10, delta=3, before=30, after=30)

    potentials = potential_waves(times, fc_allen, threshold, tol=1)
    picks = point_potentials(fc_allen, potentials)
    picks = possible_picks(picks, earthquake_loc)

    print(f'Un séisme a été détecté avec {len(picks)} débuts de phases pointées.')
    for pick in picks:
        time = times[pick]
        print(f't={time}, soit à {(starttime + time).strftime("%H:%M:%S.%f")}.')

else:
    print('Pas de séisme détecté sur ce signal.')