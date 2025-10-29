import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from typing import Tuple


def generate_dataset(n_samples: int = 1000, seed: int = 42) -> np.ndarray:
    random.seed(seed)
    study_hours = random.normal(loc=5, scale=2, size=n_samples)
    study_hours = np.clip(study_hours, 1, 12)

    sleep_hours = random.normal(loc=7, scale=1.5, size=n_samples)
    sleep_hours = np.clip(sleep_hours, 4, 10)

    quiz_scores = 40 + 9 * study_hours + 5 * sleep_hours +  random.normal(0, 5, n_samples)
    quiz_scores = np.clip(quiz_scores, 0, 100)

    attendance = 50 + 5 * study_hours + 3 * sleep_hours + random.normal(0, 8, n_samples)
    attendance = np.clip(attendance, 0, 100)
    
    data =  np.column_stack((study_hours, sleep_hours, quiz_scores, attendance))

    return data


def explore_data(data: np.ndarray) -> dict:

    stats = {}
    stats['sum'] = np.sum(data, axis=0)
    stats['mean'] = np.mean(data, axis=0)
    stats['std'] = np.std(data, axis=0)
    stats['min'] = np.min(data, axis=0)
    stats['max'] = np.max(data, axis=0)
    stats['median'] = np.median(data, axis=0)


    stats['q25'] = np.percentile(data, 25, axis=0)
    stats['q75'] = np.percentile(data, 75, axis=0)
    return stats

print(explore_data([1000000000,5000,10000,2000,3000,4000,900000,900000000])['q75'])

