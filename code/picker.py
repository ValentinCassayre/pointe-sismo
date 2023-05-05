import numpy as np


def pre_process_data(S, inventory):
    """Corrige le signal grâce au module Obspy (objets Obspy)"""
    S.attach_response(inventory)
    for T in S:
        # Conversion en signal alternatif (moyenne nulle)
        T.detrend("demean")

        # Filtrage passe-bande
        T.filter("bandpass", freqmin=2, freqmax=10)

        # Conversion l'unité du signal de counts en vitesse (optionnel)
        gain = T.meta.response.instrument_sensitivity.value
        T.data = T.data/gain


def derivate(times, data):
    """Calcul la dérivée temporelle du signal"""
    der_data = [0]
    n = len(data)

    for k in range(n-1):
        dy = (data[k+1] - data[k])
        dt = (times[k+1] - times[k])
        der_data.append(dy/dt)

    return np.array(der_data)


def derivate_np(times, data):
    """Calcul la dérivée temporelle du signal de façon optimisée"""
    n = len(data)
    der_data = np.zeros(n)

    der_data[1:] = np.diff(data)/np.diff(times)

    return der_data


def mdx_stewart(times, data, n_check=8):
    """Calcul l'enveloppe modifiée de Stewart"""
    n = len(data)
    dx = derivate(times, data)
    mdx = np.zeros(n)
    n_good = 0

    for i in range(1, n):
        if np.sign(dx[i-1]) == np.sign(dx[i]):
            n_good += 1
        else:
            n_good = 0

        if n_good > n_check:
            mdx[i] = mdx[i-1]+dx[i]
        else:
            mdx[i] = dx[i]

    return mdx


def envelope_approx(times, data):
    """Calcul l'enveloppe géométrique du signal"""
    env_times = []
    env_data = []
    n = len(data)

    for k in range(1, n-1):
        v1 = data[k] - data[k-1]
        v2 = data[k] - data[k+1]

        # ce point est un extremum local (un "pic")
        if v1 > 0 and v2 > 0 or v1 < 0 and v2 < 0:
            env_times.append(times[k])
            env_data.append(abs(data[k]))

    return np.array(env_times), np.array(env_data)


def envelope_approx_np(times, data):
    """Calcul l'enveloppe géométrique du signal de façon optimisée"""
    indices = np.nonzero(np.diff(np.sign(np.diff(data))))[0]+1

    env_times = times[indices]
    env_data = np.abs(data[indices])

    return env_times, env_data


def allen_envelope(data):
    """Calcul l'enveloppe de Allen du signal"""
    allen_data = [0]
    n = len(data)

    ci = sum(abs(x) for x in data) / \
        sum(abs(data[i]-data[i-1]) for i in range(n))

    for i in range(1, n):
        allen_data.append(data[i]**2+ci*(data[i]-data[i-1])**2)

    return np.array(allen_data)


def allen_envelope_np(data):
    """Calcul l'enveloppe de Allen du signal de façon optimisée"""
    allen_data = np.zeros(len(data))

    ci = np.sum(np.abs(data))/np.sum(np.abs(np.diff(data)))
    allen_data[1:] = data[1:]**2 + ci*np.diff(data)**2

    return allen_data


def baer_envelope(times, data):
    """Calcul l'enveloppe de Baer et Kradolfer du signal"""
    baer_data = [0]
    n = len(data)
    der_data = derivate(times, data)
    sqa_data = data**2
    sqa_der_data = der_data**2

    ci = sum(sqa_data)/sum(sqa_der_data)

    for i in range(n):
        baer_data.append(sqa_data[i]+ci*sqa_der_data[i])

    return baer_data


def baer_envelope_np(times, data):
    """Calcul l'enveloppe de Baer et Kradolfer du signal de façon optimisée"""
    sqa_data = data**2
    sqa_der_data = derivate(times, data)**2

    ci = np.sum(sqa_data)/np.sum(sqa_der_data)

    baer_data = sqa_data+ci*sqa_der_data

    return baer_data


def nstalta_from_times(times, tsta, tlta):
    """Convertit les durées des fenêtres STA et LTA en nombre de points"""
    dt = times[1] - times[0]

    nsta = int(tsta/dt)
    nlta = int(tlta/dt)

    return nsta, nlta


def stalta_allen(data, nsta, nlta):
    """Calcul la fonction caractéristique d'Allen"""
    fc = [0]*(nlta-1)
    n = len(data)

    for k in range(n-nlta+1):
        i1 = k
        i2 = k + nlta
        im = i2 - nsta

        # L'utilisation de la moyenne de Numpy réduit considérablement les calculs
        sta = np.average(data[im:i2])
        lta = np.average(data[i1:i2])

        fc.append(sta/lta)

    return fc


def stalta_allen_np(data, nsta, nlta):
    """Calcul la fonction caractéristique d'Allen de façon optimisée"""
    n = len(data)
    fc = np.zeros(n)
    sta = np.zeros(n)
    lta = np.zeros(n)

    # Calcul de la somme cummulée
    cumsum_data = np.cumsum(data)

    # Calcul des moyennes glissantes de STA et LTA à partir de la somme cummulée
    sta[nsta:] = (cumsum_data[nsta:] - cumsum_data[:-nsta])/nsta
    lta[nlta:] = (cumsum_data[nlta:] - cumsum_data[:-nlta])/nlta

    # Le rapport STA/LTA est calculé pour les termes non nuls
    indices = lta.nonzero()
    fc[indices] = sta[indices] / lta[indices]

    return fc


def stalta_recursiv(data, nsta, nlta):
    """Calcul la fonction caractéristique d'Allen de façon récursive"""
    sta = 0
    lta = 0
    fc = np.zeros(len(data))

    # Pour calculer les moyennes glissantes on doit calculer dès le 1er terme
    # même si on ne garde après que les termes après nlta
    for i, y in enumerate(data):
        sta = (y - sta)/nsta + sta
        lta = (y - lta)/nlta + lta

        if i >= nlta and lta != 0:
            fc[i] = sta / lta

    return fc


def z_swindell_snell(data, nsta):
    """Calcul la fonction caractéristique de Swindell et Snell"""
    n = len(data)
    z = np.zeros(n)
    sta = np.zeros(n)

    for k in range(nsta, n):
        sta[k] = np.average(data[k-nsta:k])

    mu = np.average(sta)
    # théorème de König-Huygens
    sqa_sigma = np.sqrt(np.average(sta**2) - mu**2)

    for k in range(nsta, n):
        if sqa_sigma != 0:
            z[k] = (sta[k] - mu)/sqa_sigma

    return z


def z_swindell_snell_np(data, nsta):
    """Calcul la fonction caractéristique de Swindell et Snell de façon optimisée"""
    n = len(data)
    fc = np.zeros(n)
    sta = np.zeros(n)

    cumsum_data = np.cumsum(data)
    sta[nsta:] = (cumsum_data[nsta:] - cumsum_data[:-nsta])/nsta

    mu = np.average(sta)
    sqa_sigma = np.sqrt(np.average(sta**2) - mu**2)

    indices = sqa_sigma.nonzero()
    fc[indices] = (sta - mu)/sqa_sigma

    return fc


def z_baer_kradolfer(data, nsta):
    """Calcul la fonction caractéristique de Baer et Kradolfer"""
    n = len(data)
    fc = np.zeros(n)

    for k in range(nsta, n):
        sqa_mu = np.average(data[k-nsta:k])**2
        sqa_sigma = np.average(data[k-nsta:k]**2) - sqa_mu
        if sqa_sigma != 0:
            fc[k] = (data[k]**2 - sqa_mu) / sqa_sigma

    return fc


def potential_waves(times, data, threshold, tol=0, delta=0, before=0, after=0):
    """Détecte un éventuel séisme à partir du signal d'une fonction caractéristique"""
    n = len(data)
    dt = times[1]-times[0]

    ntol = round(tol/dt)
    ndelta = round(delta/dt)
    nbefore = round(before/dt)
    nafter = round(after/dt)

    potentials = data > threshold

    # Ajoute une tolérence en temps pour la désactivation du dépassement du seuil
    if ntol > 0:  # évite les calculs inutiles
        i = 0
        while i < (n-ntol-1):
            if potentials[i] and not potentials[i+1]:
                for j in range(ntol):
                    if potentials[i+j]:
                        potentials[i+1:i+j] = 1
                        i += j
            i += 1

    # Vérifie s'il existe un dépassement de seuil assez long
    if ndelta > 0:
        i = 0
        while i < (n-ndelta):
            if potentials[i]:
                j = i
                while j < n and potentials[j]:
                    j += 1
                if (j-i) < ndelta:
                    potentials[i:j] = 0
                i += j
            i += 1
        potentials[i:n] = 0

    # Ajoute un contour pour sélectionner une plage plus grande
    if nbefore > 0 or nafter > 0:
        i = 1
        while i < (n-1):
            if potentials[i]:
                if not potentials[i-1]:
                    potentials[max(i-nbefore, 0):i] = 1
                if not potentials[i+1]:
                    potentials[i+1:min(i+nafter+1, n)] = 1
                    i += nafter
            i += 1

    return potentials


def point_potentials(data, potentials):
    """Pointe le début de toutes les potentielles fenêtre d'arrivée des ondes et les tries par rapport à leur maximum local"""
    above = False
    points = []
    for i in range(len(potentials)):
        if potentials[i]:  # cas d'une fenêtre
            y = data[i]
            if not above:  # pointage du début de la fenêtre
                points.append([i, y])
                above = True
            if y > points[-1][1]:  # recherche du maximum local de la fenêtre
                points[-1][1] = y
        elif above:
            above = False

    # On trie par rapport aux maximums locaux de chaque fenêtre
    points.sort(key=lambda pair: pair[1], reverse=True)
    # On ne garde que les indices de pointage
    return [pair[0] for pair in points]


def possible_picks(picks, earthquake_loc):
    """Garde les instants compris dans l'intervalle détecté"""
    return [pick for pick in picks if earthquake_loc[pick]]


def point_lmax(fc, threshold):
    """Renvoie l'index correspondant au début de l'intervalle dépassant le seuil et comprenant le maximum global"""
    threshold = 5
    imax = 0
    emax = fc[0]
    n = len(fc)
    for i in range(1, n):
        if fc[i] > emax:
            emax = fc[i]
            imax = i

    j = imax
    while j > 0 and fc[j] > threshold:
        j -= 1

    return j
