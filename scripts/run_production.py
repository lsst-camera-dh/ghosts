""" Script to run a full production (simulation and analysis)
for a set of beam and geometry configurations

"""
import logging
import pandas as pd
from ghosts.beam_configs import BASE_BEAM_SET
from ghosts import beam
from ghosts import geom
from ghosts import simulator


def run(n_geoms, prod_name='test_prod'):
    """ Run production
    """
    # Beam set
    logging.info('Generating beam configurations set')
    beam_set = BASE_BEAM_SET
    beam_set_df = beam.concat_dicts(beam_set)
    beam_set_df.head()
    # Geom set
    logging.info('Generating geometry configurations set')
    geom_set = []
    for i in range(n_geoms):
        geom_set.append(geom.build_random_geom(max_angle=0.1, max_shift=0.001))
    geom_set_df = geom.concat_dicts(geom_set)
    geom_set_df.head()

    # run simulations
    logging.info('Running and analyzing simulations')
    spots_data_frame = simulator.run_and_analyze_simulation_for_configs_sets(geom_set, beam_set)
    sims_spots_df = pd.concat(spots_data_frame)
    sims_spots_df.head()

    # merge data frames and store data
    # join spots, beam and geom data frame
    logging.info('Merging beam, spots and geom data frames')
    new_df = beam_set_df.join(sims_spots_df.set_index('beam_id'), on='beam_id', lsuffix='_beam')
    final_df = geom_set_df.join(new_df.set_index('geom_id'), on='geom_id', lsuffix='_geom')
    # attributing a prod number
    final_df.attrs['production'] = prod_name

    # save to disk
    final_df.to_parquet(f'{prod_name}.parquet')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("n_geoms")
    parser.add_argument("prod_name")
    args = parser.parse_args()
    print(args.n_geoms)
    print(args.prod_name)
    run(int(args.n_geoms), args.prod_name)
