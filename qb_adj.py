import json
import pandas as pd
import numpy
import requests
import time
import pickle
import math

df_pff_ids = pd.read_csv("D:/NFL/QB2023PreSeason/data/QB_pff_ids.csv")
df_pff_ids["Player"] = df_pff_ids["Player"].str.title()


class AirtableWrapper():
    ## This class is handles IO for an airtable base that stores
    ## starters for the current 1##
    def __init__(self, model_df, at_config, perform_starter_update=True):
        self.model_df = model_df  ## df of qbs and their meta data ##
        self.at_config = at_config  ## config for airtable including token, ids, etc ##
        ## unpack config ##
        self.base = self.at_config['base']
        self.qb_table = self.at_config['qb_table']
        self.starter_table = self.at_config['starter_table']
        self.token = self.at_config['token']
        self.qb_fields = self.at_config['qb_fields']
        self.dropdown_field_id = self.at_config['dropdown_field']
        self.base_headers = {
            'Authorization': 'Bearer {0}'.format(self.token),
            'Content-Type': 'application/json'
        }
        ## storage for various data sets and vars ##
        self.existing_qbs = None  ## qbs already written to db ##
        self.existing_qb_options = None  ## qb options in dropdown ##
        self.existing_starters = None  ## list of existing starters in AT ##
        self.starters_df = None  ## starters in AT, but in df format ##
        self.all_qbs = None  ## all qbs in model_df
        self.perform_starter_update = perform_starter_update  ## if True, update starters ##

    ## api wrapper functions ##
    def make_post_request(self, base, table, headers, data):
        ## used for creating new records ##
        ## rate limiting ##
        time.sleep(1 / 4)
        ## formulate url ##
        url = 'https://api.airtable.com/v0/{0}/{1}'.format(base, table)
        resp = requests.post(
            url,
            headers=headers,
            data=json.dumps(data)
        )

        print("Status Code:", resp.status_code)
        print("Response:", resp.text)

    def make_patch_request(self, base, table, headers, data):
        ## Used for updating existing records ##
        ## rate limiting ##
        time.sleep(1 / 4)
        ## formulate url ##
        url = 'https://api.airtable.com/v0/{0}/{1}'.format(base, table)
        resp = requests.patch(
            url,
            headers=headers,
            data=json.dumps(data)
        )
        if resp.status_code != 200:
            print('Error on patch! -- {0} -- {1}'.format(
                resp.status_code,
                resp.content
            ))

    def make_get_request(self, base, table, headers, params):
        ## used to for getting records ##
        ## rate limiting ##
        time.sleep(1 / 4)
        ## formulate url ##
        url = 'https://api.airtable.com/v0/{0}/{1}'.format(base, table)
        resp = requests.get(
            url,
            headers=headers,
            params=params
        )
        return resp

    def make_delete_request(self, base, table, headers, params):
        ## used for deleting records ##
        ## rate limiting ##
        time.sleep(1 / 4)
        ## formulate url ##
        url = 'https://api.airtable.com/v0/{0}/{1}'.format(base, table)
        requests.delete(
            url,
            headers=headers,
            params=params
        )

    def make_meta_request(self, base, headers):
        ## request schema of base ##
        ## rate limiting ##
        time.sleep(1 / 4)
        ## formulate url ##
        url = 'https://api.airtable.com/v0/meta/bases/{0}/tables'.format(base)
        resp = requests.get(
            url,
            headers=headers
        )
        return resp.json()

    def make_paginated_get(self, base, table, headers, params):
        ## make first request ##
        all_records = []
        resp = self.make_get_request(base, table, headers, params)
        records = resp.json()
        ## add records to container for initial pull ##
        print(f"records: {records}")
        for record in records['records']:
            all_records.append(record)
        ## init var loops ##
        if 'offset' in records.keys():
            offset = records['offset']
            loops = 0
        else:
            offset = None  ## if no offset, no need to paginate ##
            loops = 0
        ## loop ##
        while offset is not None and loops < 50:
            params['offset'] = offset
            resp = self.make_get_request(base, table, headers, params)
            records = resp.json()
            ## add records to container for initial pull ##
            for record in records['records']:
                all_records.append(record)
            ## update var loops ##
            if 'offset' in records.keys():
                offset = records['offset']
                loops += 1
            else:
                offset = None  ## if no offset, no need to paginate ##
                loops += 1
        ## return data ##
        return all_records

    def data_format(self, datapoint):
        ## translates a NaN to None for airtable ##
        if pd.isnull(datapoint):
            return None
        else:
            return datapoint

    def write_chunk(self, base, table, df):
        ## write chunk to airtable ##
        ## container for data to write to airtable ##
        data = {
            'records': [],
            'typecast': True
        }
        ## get table cols ##
        table_cols = df.columns.values.tolist()
        ## iterate through chunk and add to data ##
        print(f"AND HERE IS THE ATTEMPT TO PRINT TO AIRTABLE: {df.head()}")
        for index, row in df.iterrows():
            record = {
                'fields': {}
            }
            for col in table_cols:
                record['fields'][col] = self.data_format(row[col])
            ## append to date ##
            print(data.keys())
            data['records'].append(record)

        print(data['records'])
        ## write to table ##
        self.make_post_request(
            base=base,
            table=table,
            headers=self.base_headers,
            data=data
        )

    ## write chunk to airtable ##
    def update_chunk(self, base, table, df, id_col):
        ## container for data to write to airtable ##
        data = {
            'records': [],
            'typecast': True
        }
        ## get table cols ##
        table_cols = df.columns.values.tolist()
        ## iterate through chunk and add to data ##x
        for index, row in df.iterrows():
            record = {
                'id': row[id_col],
                'fields': {}
            }
            for col in table_cols:
                if col == id_col:
                    pass
                else:
                    record['fields'][col] = self.data_format(row[col])
            ## append to date ##
            data['records'].append(record)
        ## write to table ##
        self.make_patch_request(
            base=base,
            table=table,
            headers=self.base_headers,
            data=data
        )

    ## perform upsert to airtable ##
    def upsert_chunk(self, base, table, df, upsertFields, key):
        ## container for data to write to airtable ##
        data = {
            'records': [],
            'performUpsert': {
                'fieldsToMergeOn': upsertFields
            }
        }
        ## get table cols ##
        table_cols = df.columns.values.tolist()
        ## control for missing fields ##
        for field in upsertFields:
            if field not in table_cols:
                print('     {0} is not included in data. Upsert will fail...'.format(
                    field
                ))
        ## iterate through chunk and add to data ##
        for index, row in df.iterrows():
            record = {
                'fields': {}
            }
            for col in table_cols:
                record['fields'][col] = self.data_format(row[col])
            ## append to date ##
            data['records'].append(record)
        ## write to table ##
        self.make_patch_request(
            base=base,
            table=table,
            headers={
                'Authorization': 'Bearer {0}'.format(key),
                'Content-Type': 'application/json'
            },
            data=data
        )

    ## break df into chunks of 10 and write to airtable ##
    def write_table(self, base, table, df):
        ## break df into chunks of 10 ##
        ## determine size of df ##
        df_len = len(df)
        chunks_needed = math.ceil(df_len / 10)
        ## split ##
        df_chunks = numpy.array_split(df, chunks_needed)
        ## write ##
        for chunk in df_chunks:
            ## turn chunk into record ##
            self.write_chunk(base, table, chunk)

    ## break df into chunks of 10 and write to airtable ##
    def update_table(self, base, table, df, id_col):
        ## break df into chunks of 10 ##
        ## determine size of df ##
        df_len = len(df)
        chunks_needed = math.ceil(df_len / 10)
        ## split ##
        df_chunks = numpy.array_split(df, chunks_needed)
        ## write ##
        for chunk in df_chunks:
            ## turn chunk into record ##
            self.update_chunk(base, table, chunk, id_col)

    ## fucntional abstractions for wrapper ##
    def get_existing_qbs(self):
        ## gets existing QBs from airtable ##
        ## get existing qbs ##
        qbs_resp = self.make_paginated_get(
            base=self.base,
            table=self.qb_table,
            headers=self.base_headers,
            params={
                ##'fields' : [self.qb_fields]
            }
        )
        ## container for qbs ##
        qbs = []
        ## iterate through qb response and add to container ##

        for qb in qbs_resp:
            print(qb['fields']['player_id'])
            qbs.append(qb['fields']['player_id'])
        ## return ##
        self.existing_qbs = qbs

    def get_qb_options(self):
        ## gets a list of QBs that are options in the drop down ##
        ## get base schema ##
        base_schema = self.make_meta_request(
            base=self.base,
            headers=self.base_headers,
        )
        ## parse ##
        options = []
        for table in base_schema['tables']:
            if table['id'] == self.starter_table:
                for field in table['fields']:
                    if field['id'] == self.dropdown_field_id:
                        for option in field['options']['choices']:
                            options.append(option['name'])
        ## return ##
        self.qb_options = options

    def get_starters(self):
        ## gets existing QBs from airtable ##
        ## get existing qbs ##
        qbs_resp = self.make_paginated_get(
            base=self.base,
            table=self.starter_table,
            headers=self.base_headers,
            params={
                ##'fields' : [self.qb_fields]
            }
        )
        ## structure for existing starters, which has a key of the team ##
        ## and values of record id and qb_id ##
        existing_starters = {}


        for record in qbs_resp:
            print(record)
            if 'fields' in record and 'team' in record['fields'] and 'player_id' in record['fields']:
                existing_starters[record['fields']['team']] = {
                    'record_id': record['id'],
                    'qb_id': record['fields']['player_id'][0]
                }
            else:
                print("Record with id {} does not have the required fields.".format(record['id']))
        ## write ##
        self.existing_starters = existing_starters

    def write_qbs(self, qbs_to_write):
        ## write a df containing qb meta to the qb db in airtable ##
        self.write_table(
            base=self.base,
            table=self.qb_table,
            df=qbs_to_write
        )

    def write_qb_options(self, qb_options_to_write):
        ## to update an option to the dropdown, you need to create a record ##
        ## with typecase set to true ##
        ## to do this, loop through new options. On the first, create a dummary record ##
        ## on subsequents records, upsert that record ##
        ## on the final, delete the dummy record ##
        ## container for dummy record id ##
        dummy_id = None
        print(f"QB options to write:{qb_options_to_write}")
        for index, value in enumerate(qb_options_to_write):
            ## create record structure ##
            data = {
                'records': [
                    {
                        'fields': {
                            'team': 'DUMMY',
                            'qb_id': value
                        }
                    }
                ],
                'typecast': True
            }
            if index == 0:
                ## if first record, create the dummy ##
                self.make_post_request(
                    base=self.base,
                    table=self.starter_table,
                    headers=self.base_headers,
                    data=data
                )
                ## retrieve record to get id ##
                resp = self.make_get_request(
                    base=self.base,
                    table=self.starter_table,
                    headers=self.base_headers,
                    params={
                        'filterByFormula': 'team = "DUMMY"'
                    }
                )
                resp = resp.json()
                print(resp)
                print(resp['records'])
                dummy_id = resp['records'][0]['id']

            else:
                ## update record with dummy id ##
                data['records'][0]['id'] = dummy_id
                ## make a patch request ##
                self.make_patch_request(
                    base=self.base,
                    table=self.starter_table,
                    headers=self.base_headers,
                    data=data
                )
            ## if last record, delete dummy ##
            if index == len(qb_options_to_write) - 1:
                r = self.make_delete_request(
                    base=self.base,
                    table=self.starter_table,
                    headers=self.base_headers,
                    params={
                        'records[]': dummy_id
                    }
                )

    ## model functions ##
    def get_qbs(self):
        ## gets a unique set of QBs from the data file ##
        ## note, this only stores QBs that have made a start ##
        qbs = self.model_df.copy()
        ## get most recent ##
        qbs = qbs.sort_values(
            by=['gameday'],
            ascending=[False]
        ).reset_index(drop=True)
        ## add a field that combines id and display name ##
        qbs['qb_id'] = qbs['player_display_name'] + '-' + qbs['player_id'].fillna(0).round(0).astype(int).astype(str)

        print("QBs columns are below:")


        qbs = qbs[[
            'qb_id', 'player_id', 'player_display_name','team',
            'start_number', 'rookie_year', 'entry_year',
            'draft_number'
        ]].groupby(['player_id']).head(1)
        ## return ##


        qbs['player_id'] = qbs['player_id'].astype(int)
        qbs['start_number'] = qbs['start_number'].astype(int)
        qbs['rookie_year'] = qbs['rookie_year'].astype(int)
        qbs['entry_year'] = qbs['entry_year'].astype(int)
        qbs['draft_number'] = qbs['draft_number'].astype('Int64')

        self.all_qbs = qbs



    def get_last_starter(self):
        ## for each team, determines last starter, which is assumed ##
        ## to be the starter for the next week ##
        print('Getting last starters...')
        starters = self.model_df.copy()
        starters = starters.sort_values(
            by=['gameday'],
            ascending=[False]
        ).reset_index(drop=True)
        ## add a field that combines id and display name ##
        starters['qb_id'] = starters['player_display_name'] + '-' + starters['player_id'].fillna(0).round(0).astype(int).astype(str)
        starters = starters[[
            'team', 'qb_id',
        ]].groupby(['team']).head(1)
        ## return ##
        return starters

    ## actual functions that get called ##
    def update_qb_table(self):
        ## checks qbs in airtable against qbs in data ##
        ## updates the delta ##
        print('Updating QB table...')
        ## get existing qbs ##
        self.get_existing_qbs()
        print(self.existing_qbs)
        ## get qbs from data ##
        self.get_qbs()
        ## get delta ##
        delta = self.all_qbs[
            ~numpy.isin(
                self.all_qbs['player_id'],
                self.existing_qbs
            )
        ].copy()



        delta = delta[['qb_id', 'player_id', 'player_display_name','team', 'start_number',
       'rookie_year', 'draft_number']]

        delta.columns = ['qb_id', 'player_id', 'player_name','team' ,'start_number','rookie_year', 'draft_number']

        ## determine write ##
        if len(delta) > 0:
            print('     Found {0} new QBs'.format(len(delta)))
            ## write ##
            self.write_qbs(delta)
            ## update existing qbs so its accurate ##
            for qb in delta['player_id'].unique().tolist():
                self.existing_qbs.append(qb)
        else:
            print('     No new QBs needed')

    def update_qb_table_games_started(self):


        data_qbs=self.all_qbs.copy()

        data_qbs.to_csv('D:/NFL/NFL Beyond/data_qbs.csv')

        data_qbs_dict = data_qbs.set_index('player_id')['start_number'].to_dict()

        # Update function
        def update_start_number(player_id, start_number,base, starter_table, headers):
            # First, fetch the record ID using the player_id from Airtable
            time.sleep(1 / 4)
            url = 'https://api.airtable.com/v0/{0}/{1}'.format(base, starter_table)
            response = requests.get(
                url,


                headers=headers,
                params={"filterByFormula": f"{{player_id}} = '{player_id}'"}
            )


            records = response.json().get('records')

            if not records:
                print(f"No record found for player_id: {player_id}")
                return

            record_id = records[0]['id']

            # Update the start_number using the record ID
            response = requests.patch(
                f"{url}/{record_id}",

                headers=headers,
                json={"fields": {"start_number": start_number}}
            )

            if response.status_code == 200:
                print(f"Updated start_number for player_id: {player_id}")
            else:
                print(
                    f"Failed to update player_id: {player_id}. Status code: {response.status_code}, Message: {response.text}")

        # Iterate over the dictionary to update each player's start_number
        for player_id, start_number in data_qbs_dict.items():
            update_start_number(player_id, start_number,self.base, self.qb_table, self.base_headers)



    def update_qb_options(self):
        ## updates the QB option dropdown to reflect QBs in the ##
        ## database ##
        print('Updating QB options...')
        ## update existing options ##
        self.get_qb_options()
        ## determine all values that should be in dropdown ##
        delta = self.all_qbs[
            ~numpy.isin(
                self.all_qbs['qb_id'],
                self.qb_options
            )
        ].copy()

        print(f"delta is {delta.reset_index(drop=True)}")

        print(f"qb_options is {self.qb_options}")
        ## determine write ##
        if len(delta) > 0:
            print('     Found {0} new QB options'.format(len(delta)))
            ## write ##
            # self.write_qb_options(delta['qb_id'].unique().tolist())
        else:
            print('     No new QB options needed')

    def update_starters(self):


        if not self.perform_starter_update:
            return
        ## reads the starter table in airtable and determines ##
        ## if any starters are different from the previous week ##
        print('Updating starters...')
        ## get last week's starters from AT ##
        self.get_starters()
        existing_starters = self.existing_starters
        ## get this weeks starters from data #
        this_weeks_starters = self.get_last_starter().reset_index(drop=True)
        ## structure for holding updates ##
        writes = []
        updates = []
        print(f"Last weeks starters are: {this_weeks_starters}")
        ## loop through teams ##
        for index, row in this_weeks_starters.iterrows():
            ## get team ##
            team = row['team']
            if team in existing_starters:
                ## if team is in the AT table (it should be) check starter ##
                if existing_starters[team]['qb_id'] != row['qb_id']:
                    ## if starter is not match, create update rec ##
                    updates.append({
                        'id': existing_starters[team]['record_id'],
                        'qb_id': row['qb_id'],
                        ## airtable automations dont trigger on API update, so ##
                        ## zero out the fields so it's obvious they need to be updated ##
                        'start_number': numpy.nan,
                        'rookie_year': numpy.nan,
                        'entry_year': numpy.nan,
                        'draft_number': numpy.nan,
                        'player_display_name': numpy.nan,
                        'player_id': numpy.nan
                    })
            else:
                ## if team is not in the AT table, create write rec ##
                writes.append({
                    'team': team,
                    'qb_id': row['qb_id']
                })
        ## write if necessary ##
        if len(writes) > 0:
            print('     Found {0} new teams'.format(len(writes)))
            self.write_table(
                base=self.base,
                table=self.starter_table,
                df=pd.DataFrame(writes)
            )
        ## update if necessary ##
        if len(updates) > 0:
            print('     Found {0} updated starters'.format(len(updates)))
            self.update_table(
                base=self.base,
                table=self.starter_table,
                df=pd.DataFrame(updates),
                id_col='id'
            )

    def pull_current_starters(self):
        ## pulls the current starters from the airtable ##
        ## and stores as a DF for the elo constructor ##
        qbs_resp = self.make_paginated_get(
            base=self.base,
            table=self.starter_table,
            headers=self.base_headers,
            params={
                ##'fields' : self.qb_fields
            }
        )
        ## structure for existing starters, which has a key of the team ##
        ## and values of record id and qb_id ##
        starters_data = []
        for record in qbs_resp:

            ## control for missing ##
            # for field in ['team', 'player_id', 'player_display_name', 'draft_number']:
                # if field not in record['fields']:
                #     record['fields'][field] = numpy.nan


            player_display_name = record['fields']['player_name'][0] if isinstance(record['fields']['player_name'], list) else record['fields']['player_name']
            player_id = record['fields']['player_id'][0] if isinstance(record['fields']['player_id'],
                                                                             list) else record['fields']['player_id']

            draft_number = record['fields']['draft_number'][0] if isinstance(record['fields']['draft_number'], list) else record['fields']['draft_number']

            start_number = record['fields']['start_number'][0] if isinstance(record['fields']['start_number'], list) else record['fields']['start_number']

            starters_data.append({
                'team': record['fields']['team'],
                'player_id': player_id,
                'player_display_name': player_display_name,
                'draft_number': draft_number,
                'start_number': start_number
            })
        ## write ##
        self.starters_df = pd.DataFrame(starters_data)


class QBModel():
    ## This class is used to store, retrieve, and update data as we
    ## iterate over the game file ##
    def __init__(self, games, model_config):
        self.games = games
        self.config = model_config
        self.season_avgs = {}  ## storage for season averages ##
        self.team_avgs = {}  ## storage for season team averages ##
        self.qbs = {}  ## storage for most recent QB Data
        self.teams = {}  ## storage for most recent team data
        self.data = []  ## storage for all game records ##
        self.league_avg_def = model_config[
            'init_value']  ## each week, we need to calculate the league avg def for adjs ##
        self.current_week = 1 ## track the current week to know when it has changed ##
        self.model_run_time = 0  ## track the time it takes to run the model ##
        ## import original elo file location ##
        data_folder = "D:/NFL/QB2023PreSeason/data"
        self.original_file_loc = '{0}/original_elo_file.csv'.format(data_folder)
        ## initial ##
        self.chrono_sort()  ## sort by data, so games can be iter'd ##
        self.add_averages()

    ## setup functions ##
    def chrono_sort(self):
        ## sort games by date ##
        self.games = self.games.sort_values(
            by=['season', 'week', 'game_id'],
            ascending=[True, True, True]
        ).reset_index(drop=True)

    def add_averages(self):
        ## adds the avg QB values for teams and leagues which are used in reversion ##
        ## calc team averages ##
        team_avgs = self.games.groupby(
            ['season', 'team']
        )['team_VALUE'].mean().reset_index()
        ## calc league average ##
        season_avgs = self.games.groupby(
            ['season']
        )['team_VALUE'].mean().reset_index()
        ## write to stoarge ##
        for index, row in team_avgs.iterrows():
            self.team_avgs['{0}{1}'.format(row['season'], row['team'])] = row['team_VALUE']
        for index, row in season_avgs.iterrows():
            self.season_avgs[row['season']] = row['team_VALUE']

    ## retrieval functions for averages ##
    def get_prev_season_team_avg(self, season, team):
        ## get the teams previous season average while controlling for errors ##
        ## the first season will not have a pervous season ##
        return self.team_avgs.get('{0}{1}'.format(
            season - 1, team
        ), self.config['init_value'])

    def get_prev_season_league_avg(self, season):
        ## get the leagues previous season average while controlling for errors ##
        ## the first season will not have a pervous season ##
        return self.season_avgs.get(season - 1, self.config['init_value'])

    ## retrieval functions for QB ##
    def init_qb(self, qb_id, season, team, draft_number, gameday):
        ## initialize qb into storage while also calculating their initial value ##
        ## round is blank (ie undrafted) enter high value ##
        if pd.isnull(draft_number):
            draft_number = self.config['rookie_undrafted_draft_number']
        else:
            draft_number = draft_number[0] if isinstance(draft_number, list) else draft_number

        ## get the previous season averages for the team and league
        prev_season_team_avg = self.get_prev_season_team_avg(season, team)
        prev_season_league_avg = self.get_prev_season_league_avg(season)
        ## calculate the initial value ##





        init_value = min(
            ## value over team's previous average based on draft number ##
            (self.config['rookie_draft_intercept'] + (self.config['rookie_draft_slope'] * math.log(draft_number))) +
            ## team value is regressed to the league average ##
            (
                    ((1 - self.config['rookie_league_reg']) * prev_season_team_avg) +
                    (self.config['rookie_league_reg'] * prev_season_league_avg)
            ),
            ## value is capped at a discount of the league average ##
            ((1 + self.config['rookie_league_cap']) * prev_season_league_avg)
        )
        ## write to storage ##


        try:
            self.qbs[qb_id] = {
                'current_value': init_value,
                'current_variance': init_value,
                'rolling_value': init_value,
                'starts': 0,
                'season_starts': 0,
                'first_game_date': gameday,
                'first_game_season': season,
                'last_game_date': None,
                'last_game_season': None
            }
        except:
            print(self.qbs)

    def s_curve(self, height, mp, x, direction='down'):
        ## calculate s-curve, which are used for progression discounting and multiplying ##
        if direction == 'down':
            return (
                    1 - (1 / (1 + 1.5 ** (
                    (-1 * (x - mp)) *
                    (10 / mp)
            )))
            ) * height
        else:
            return (1 - (
                    1 - (1 / (1 + 1.5 ** (
                    (-1 * (x - mp)) *
                    (10 / mp)
            )))
            )) * height

    def handle_qb_regression(self, qb, season):
        ## regress qb to the league average ##
        ## first, get the previous season average ##
        prev_season_league_avg = self.get_prev_season_league_avg(season)
        ## determine regression amounts based on model curves ##
        league_regression = self.s_curve(
            self.config['player_regression_league_height'],
            self.config['player_regression_league_mp'],
            qb['starts'],
            'down'
        )
        career_regression = self.s_curve(
            self.config['player_regression_career_height'],
            self.config['player_regression_career_mp'],
            qb['starts'],
            'up'
        )
        ## calculate the new value ##
        ## if the qb didnt play much the previous (ie was a backup) this is ##
        ## signal that they are not league average quality ##
        ## In this case, we discount the league average regression portion ##
        league_regression = (
                league_regression *
                self.s_curve(
                    1,
                    4,
                    qb['season_starts'],
                    'up'
                )
        )
        ## normalize the combined career and league regression to not exceed 100% ##
        total_regression = league_regression + career_regression
        if total_regression > 1:
            league_regression = league_regression / total_regression
            career_regression = career_regression / total_regression
        ## calculate value ##
        qb['current_value'] = (
                (1 - league_regression - career_regression) * qb['current_value'] +
                (league_regression * prev_season_league_avg) +
                (career_regression * qb['rolling_value'])
        )
        ## update season ##
        ## return the qb object ##
        return qb

    def get_qb_value(self, row):
        ## retrieve the current value of the qb before the game ##
        ## this takes the entire row as we may need to unpack values to send to
        ## other models ##
        ## get qb from storage ##

        player_id = row['player_id'][0] if isinstance(row['player_id'], list) else row['player_id']

        if player_id not in self.qbs:
            self.init_qb(
                player_id, row['season'], row['team'],
                row['draft_number'], row['gameday']
            )
        qb = self.qbs[player_id]
        ## handle regression ##
        if qb['last_game_season'] is None:
            pass
        elif row['season'] > qb['last_game_season']:
            qb = self.handle_qb_regression(qb, row['season'])
        ## return value ##
        return qb['current_value']

    def update_qb_value(self, qb_id, value, gameday, season):
        ## to speed up, minimize the number of times we lookup from storage ##
        qb_ = self.qbs[qb_id]
        ## first store the pre-update value, which i sneeded for rolling variance ##
        old_value = qb_['current_value']
        ## update the qb value after the game ##
        qb_['current_value'] = (
                self.config['player_sf'] * value +
                (1 - self.config['player_sf']) * qb_['current_value']
        )
        ## for rolling value, use a progressively deweighted ewma ##
        ## set rolling sf ##
        rolling_sf = (
                self.config['player_career_sf_base'] +
                self.s_curve(
                    self.config['player_career_sf_height'],
                    self.config['player_career_sf_mp'],
                    qb_['starts'],
                    'down'
                )
        )
        qb_['rolling_value'] = (
                rolling_sf * value +
                (1 - rolling_sf) * qb_['rolling_value']
        )
        ## update variance ##
        ## 𝜎2𝑛=(1−𝛼)𝜎2𝑛−1+𝛼(𝑥𝑛−𝜇𝑛−1)(𝑥𝑛−𝜇𝑛) ##
        ## https://stats.stackexchange.com/questions/6874/exponential-weighted-moving-skewness-kurtosis ##
        qb_['current_variance'] = (
                self.config['player_sf'] * (value - old_value) * (value - qb_['current_value']) +
                (1 - self.config['player_sf']) * qb_['current_variance']
        )
        ## update meta ##
        qb_['starts'] += 1
        qb_['season_starts'] += 1
        qb_['last_game_date'] = gameday
        qb_['last_game_season'] = season
        ## write back to storage ##
        self.qbs[qb_id] = qb_
        ## return updated value ##
        return qb_['current_value']

    ## function for initing teams ##
    def init_team(self, team):
        ## initialize team into storage ##
        self.teams[team] = {
            'off_value': self.config['init_value'],
            'def_value': self.config['init_value']
        }
        ## return the team object ##
        return self.teams[team]

    ## functions for getting team def values ##
    def update_league_avg_def(self):
        ## take the average of all team defensive scores ##
        ## use this to update the league average variable ##
        defensive_values = []
        for team, val in self.teams.items():
            defensive_values.append(val['def_value'])
        ## update ##
        self.league_avg_def = numpy.mean(defensive_values)

    def handle_team_def_regression(self, team_obj):
        ## simple func for regressing team values to mean
        team_obj['def_value'] = (
                (1 - self.config['team_def_reversion']) * team_obj['def_value'] +
                self.config['team_def_reversion'] * self.league_avg_def
        )
        return team_obj

    def get_team_def_value(self, team, week):
        ## get the defensive value of the team ##
        ## if the week has changed, update the league average ##
        if self.current_week != week:
            self.update_league_avg_def()
            self.current_week = week
        ## retrieve the team from storage ##
        if team not in self.teams:
            self.init_team(team)
        team_obj = self.teams[team]
        ## regress if needed ##
        if week == 1:
            team_obj = self.handle_team_def_regression(team_obj)
            ## update in db since this value has now changed ##
            self.teams[team]['def_value'] = team_obj['def_value']
        ## return the teams def value and val vs league ##
        return team_obj['def_value'], team_obj['def_value'] - self.league_avg_def

    def update_team_def_value(self, team, value):
        ## update the team value after the game ##
        self.teams[team]['def_value'] = (
                self.config['team_def_sf'] * value +
                (1 - self.config['team_def_sf']) * self.teams[team]['def_value']
        )
        ## return updated value ##
        return self.teams[team]['def_value']

    ## functions for getting team off values ##
    def handle_team_off_regression(self, team_obj, qb_val, season):
        ## simple func for regressing team values to the week 1 starter value ##
        team_obj['off_value'] = (
                (1 - self.config['team_off_league_reversion'] - self.config['team_off_qb_reversion']) * team_obj[
            'off_value'] +
                self.config['team_off_qb_reversion'] * qb_val +
                self.config['team_off_league_reversion'] * self.get_prev_season_league_avg(season)
        )
        return team_obj

    def get_team_off_value(self, team, qb_val, season):
        ## function for getting the offensive value of the team ##
        ## retrieve the team from storage ##
        if team not in self.teams:
            self.init_team(team)
        team_obj = self.teams[team]
        ## handle offensive regression as needed ##
        if self.current_week == 1:
            team_obj = self.handle_team_off_regression(team_obj, qb_val, season)
            ## update in db since this value has now changed ##
            self.teams[team]['off_value'] = team_obj['off_value']
        ## return off value and adj relative to qb ##
        return team_obj['off_value'], qb_val - team_obj['off_value']

    def update_team_off_value(self, team, value):
        ## update the team value after the game ##
        self.teams[team]['off_value'] = (
                self.config['team_off_sf'] * value +
                (1 - self.config['team_off_sf']) * self.teams[team]['off_value']
        )
        ## return updated value ##
        return self.teams[team]['off_value']

    ## model functions ##
    def run_model(self):
        ## function that iters through games df and runs model ##
        ## set a start epoch time ##
        start_time = time.time()
        ## clear out any existing values ##
        self.qbs = {}  ## storage for most recent QB Data
        self.teams = {}  ## storage for most recent team data
        self.data = []  ## storage for all game records ##
        self.league_avg_def = self.config[
            'init_value']  ## each week, we need to calculate the league avg def for adjs ##
        self.current_week = 1  ## track the current week to know when it has changed ##
        ## iterate through games df ##

        self.games.to_csv("D:/NFL/QB2023PreSeason/Nolan_test.csv")
        self.games = self.games.drop_duplicates()

        for index, row in self.games.iterrows():
            ## get qb value ##
            qb_val = self.get_qb_value(row)
            ## get team def value ##
            team_def_val, team_def_adj = self.get_team_def_value(row['opponent'], row['week'])
            ## get team off value ##
            team_off_val, team_off_adj = self.get_team_off_value(row['team'], qb_val, row['season'])
            ## calc qb adj ##
            qb_adj = qb_val - team_off_val
            ## calc adjusted game value
            adj_val = row['player_VALUE'] - team_def_adj
            ## update qb value ##
            self.update_qb_value(row['player_id'], adj_val, row['gameday'], row['season'])
            ## update team def value ##
            self.update_team_def_value(row['opponent'], row['player_VALUE'])
            ## update team off value ##
            self.update_team_off_value(row['team'], adj_val)
            ## add all values to the row ##
            row['qb_value_pre'] = qb_val
            row['team_value_pre'] = team_off_val
            row['qb_adj'] = qb_adj
            row['opponent_def_value_pre'] = team_def_val
            row['opponent_def_adj'] = team_def_adj
            row['player_VALUE_adj'] = adj_val
            row['qb_value_post'] = self.qbs[row['player_id']]['current_value']
            row['team_value_post'] = self.teams[row['team']]['off_value']
            row['opponent_def_value_post'] = self.teams[row['opponent']]['def_value']
            ## write row to data ##
            self.data.append(row)
        end_time = time.time()
        self.model_runtime = end_time - start_time

    ## scoring ##
    def add_elo(self, df):
        ## add elo values from 538 to df for comparison of accuracy ##
        ## read in elo data ##
        elo = pd.read_csv(
            self.original_file_loc,
            index_col=0,
        )
        ## flatten elo df ##
        elo = pd.concat([
            elo[[
                'date', 'team1', 'team2', 'qb1', 'qb1_value_pre', 'qb1_adj'
            ]].rename(columns={
                'date': 'gameday',
                'team1': 'team',
                'team2': 'opponent',
                'qb1': 'player_display_name',
                'qb1_value_pre': 'f38_projected_value',
                'qb1_adj': 'f38_team_adj'
            }),
            elo[[
                'date', 'team2', 'team1', 'qb2', 'qb2_value_pre', 'qb2_adj'
            ]].rename(columns={
                'date': 'gameday',
                'team2': 'team',
                'team1': 'opponent',
                'qb2': 'player_display_name',
                'qb2_value_pre': 'f38_projected_value',
                'qb2_adj': 'f38_team_adj'
            }),
        ])
        ## dedupe ##
        elo = elo.groupby([
            'gameday', 'team', 'opponent', 'player_display_name'
        ]).head(1)
        ## convert elo to value ##
        elo['f38_projected_value'] = elo['f38_projected_value'] / 3.3
        elo['f38_team_adj'] = elo['f38_team_adj'] / 3.3
        ## merge elo data ##
        df = pd.merge(
            df,
            elo,
            on=['gameday', 'team', 'opponent', 'player_display_name'],
            how='left'
        )
        ## return df ##
        return df

    def score_model(self, first_season=2009, add_elo=True):
        ## function for scoring model for testing purposes ##
        ## create df from data ##
        df = pd.DataFrame(self.data)
        ## get mean squared error ##
        df['se'] = (df['qb_value_pre'] - df['player_VALUE_adj']) ** 2
        df['abs_error'] = numpy.absolute(df['qb_value_pre'] - df['player_VALUE_adj'])
        ## only look at data past first season ##
        ## this is to give model time to catch up since we are starting in 1999 ##
        ## and veteran QBs are treated like rookies in that season ##
        df = df[df['season'] >= first_season].copy()
        ## copy config to serve as a record of what was used ##
        record = self.config.copy()
        ## add rmse and mae to record ##
        record['rmse'] = df['se'].mean() ** 0.5
        record['mae'] = df['abs_error'].mean()
        if add_elo:
            ## add elo data ##
            df = self.add_elo(df)
            ## add comparison to 538 ##
            f = df[~pd.isnull(df['f38_projected_value'])].copy()
            f['f38_se'] = (f['f38_projected_value'] - f['player_VALUE_adj']) ** 2
            record['delta_vs_538'] = (f['f38_se'].mean() ** 0.5) - (f['se'].mean() ** 0.5)
            ## add rookie model comp ##
            r = f[f['start_number'] <= 10].copy()
            record['delta_vs_538_rookies'] = (r['f38_se'].mean() ** 0.5) - (r['se'].mean() ** 0.5)
        record['model_runtime'] = self.model_runtime
        ## return record ##
        return record

    def score_adj(self, first_season=2009):
        ## Function for scoring the team adjustment ##
        ## While the model should try to predict VALUE as best as it can ##
        ## The team adj should try to get as close to the 538 team adj as this ##
        ## ghe main nfelo model has already been optimized for this value ##
        ## create df from data ##
        df = pd.DataFrame(self.data)
        df = df[df['season'] >= first_season].copy()
        ## add elo ##
        df = self.add_elo(df)
        ## add comparison to 538 ##
        f = df[~pd.isnull(df['f38_team_adj'])].copy()
        f['adj_se'] = (f['f38_team_adj'] - f['qb_adj']) ** 2
        ## add rmse to record ##
        record = self.config.copy()
        record['adjustment_rmse'] = f['adj_se'].mean() ** 0.5
        return record


class DataLoader():
    ## this class retrieves, loads, and merges data ##
    def __init__(self):
        ## dfs we want to output ##
        self.model_df = None
        self.games_df = None
        ## locations of external data ##
        self.player_stats_url = 'https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats.csv.gz'
        self.player_info_url = 'https://github.com/nflverse/nflverse-data/releases/download/players/players.csv'
        self.game_data_url = 'https://github.com/nflverse/nfldata/raw/master/data/games.csv'
        ## repl dicts ##
        self.player_file_repl = {
            'LA': 'LAR',
            'LV': 'OAK',
        }
        self.games_file_repl = {
            'LA': 'LAR',
            'LV': 'OAK',
            'STL': 'LAR',
            'SD': 'LAC',
        }
        ## variabbles
        self.stat_cols = [
            'completions', 'attempts', 'passing_yards', 'passing_tds',
            'interceptions', 'sacks', 'carries', 'rushing_yards', 'rushing_tds'
        ]
        ## load file path ##
        data_folder = "D:/NFL/QB2023PreSeason/data"
        self.missing_draft_data = '{0}/missing_draft_data.csv'.format(data_folder)
        ## get data on load ##
        self.pull_data()

    def retrieve_player_stats(self):
        ## download stats from nflverse ##
        ## these are pre-aggregated from the pbp, which saves time/compute ##
        try:
            df = pd.read_csv(
                self.player_stats_url,
                compression='gzip',
                low_memory=False
            )
            ## only  qbs are relevant ##
            df = df[
                df['position'] == 'QB'
                ].copy()
            ## file only has recent abbreviation of team name. Standardize so we can join ##
            df['recent_team'] = df['recent_team'].replace(self.player_file_repl)
            df = df.rename(columns={
                'recent_team': 'team',
            })
            return df
        except Exception as e:
            print('     Error retrieving player stats: ' + str(e))
            return None

    def retrieve_player_meta(self, df):
        ## get player meta and add it to the stats ##
        ## will be used for draft position and joining ##
        ## the meta file has missing draft data, which has been manually compiled ##
        ## and will be added ##
        def add_missing_draft_data(df):
            ## load missing draft data ##
            missing_draft = pd.read_csv(
                self.missing_draft_data,
                index_col=0
            )
            ## groupby id to ensure no dupes ##
            missing_draft = missing_draft.groupby(['player_id']).head(1)
            ## rename the cols, which will fill if main in NA ##
            missing_draft = missing_draft.rename(columns={
                'rookie_year': 'rookie_year_fill',
                'draft_number': 'draft_number_fill',
                'entry_year': 'entry_year_fill',
                'birth_date': 'birth_date_fill',
            })
            ## add to data ##
            df = pd.merge(
                df,
                missing_draft[[
                    'player_id', 'rookie_year_fill', 'draft_number_fill',
                    'entry_year_fill', 'birth_date_fill'
                ]],
                on=['player_id'],
                how='left'
            )
            ## fill in missing data ##
            for col in [
                'rookie_year', 'draft_number', 'entry_year', 'birth_date'
            ]:
                ## fill in missing data ##
                df[col] = df[col].combine_first(df[col + '_fill'])
                ## and then drop fill col ##
                df = df.drop(columns=[col + '_fill'])
            ## return ##
            return df

        ## get player meta ##
        try:
            meta = pd.read_csv(
                self.player_info_url
            )
            meta = meta.groupby(['gsis_id']).head(1)
            ## add to df ##
            df = pd.merge(
                df,
                meta[[
                    'gsis_id', 'first_name', 'last_name',
                    'birth_date', 'rookie_year', 'entry_year',
                    'draft_number'
                ]].rename(columns={
                    'gsis_id': 'player_id',
                }),
                on=['player_id'],
                how='left'
            )
            ## add missing draft data ##
            df = add_missing_draft_data(df)
            ## return ##
            return df
        except Exception as e:
            print('     Error retrieving player info: ' + str(e))
            return None

    def add_game_data(self, df):
        ## add game data ##
        try:
            game = pd.read_csv(
                self.game_data_url
            )
            ## replace team names ##
            game['home_team'] = game['home_team'].replace(self.games_file_repl)
            game['away_team'] = game['away_team'].replace(self.games_file_repl)
            ## use replaced home and away names to reconstitute the game id ##
            game['game_id'] = (
                    game['season'].astype(str) + '_' +
                    game['week'].astype(str).str.zfill(2) + '_' +
                    game['away_team'] + '_' +
                    game['home_team']
            )
            ## games will be used in the future so add to class ##
            self.games = game.copy()
            ## flatten ##
            game_flat = pd.concat([
                game[[
                    'game_id', 'gameday', 'season', 'week',
                    'home_team', 'away_team',
                    'home_qb_id', 'home_qb_name',
                    'away_qb_id', 'away_qb_name',
                ]].rename(columns={
                    'home_team': 'team',
                    'home_qb_id': 'starter_id',
                    'home_qb_name': 'starter_name',
                    'away_team': 'opponent',
                    'away_qb_id': 'opponent_starter_id',
                    'away_qb_name': 'opponent_starter_name',
                }),
                game[[
                    'game_id', 'gameday', 'season', 'week',
                    'home_team', 'away_team',
                    'home_qb_id', 'home_qb_name',
                    'away_qb_id', 'away_qb_name',
                ]].rename(columns={
                    'away_team': 'team',
                    'away_qb_id': 'starter_id',
                    'away_qb_name': 'starter_name',
                    'home_team': 'opponent',
                    'home_qb_id': 'opponent_starter_id',
                    'home_qb_name': 'opponent_starter_name',
                })
            ])
            ## add to df ##
            df = pd.merge(
                df,
                game_flat,
                on=['season', 'week', 'team'],
                how='left'
            )
            ## return ##
            return df
        except Exception as e:
            print('     Error adding game data: ' + str(e))
            return None

    ## funcs for calculating value and formatting model file ##
    def aggregate_team_stats(self, df, team_field='team'):
        ## aggregates the individual player file into a team file ##
        ## team field denotes whether to use team or opponent ##
        return df.groupby(['game_id', 'season', 'week', 'gameday', team_field]).agg(
            completions=('completions', 'sum'),
            attempts=('attempts', 'sum'),
            passing_yards=('passing_yards', 'sum'),
            passing_tds=('passing_tds', 'sum'),
            interceptions=('interceptions', 'sum'),
            sacks=('sacks', 'sum'),
            carries=('carries', 'sum'),
            rushing_yards=('rushing_yards', 'sum'),
            rushing_tds=('rushing_tds', 'sum'),
        ).reset_index().rename(columns={
            team_field: 'team'
        })

    def iso_top_passer(self, df):
        ## So as not to update the rating of a QB who had few passes, only include
        ## the top passer ##
        ## however, if this player was not the starter, then we need to override ##
        ## add starter info ##
        ## this needs to be cleaned up -- i think the attempts are not relevant as we are just using starter
        df['is_starter'] = numpy.where(
            df['player_id'] == df['starter_id'],
            1,
            numpy.nan
        )
        return df.sort_values(
            by=['game_id', 'is_starter', 'attempts'],
            ascending=[True, False, False]
        ).groupby(['game_id', 'team']).head(1).reset_index(drop=True)

    def format_top_passer(self, df):
        ## add the start number to the top passer and get rid of unecessary fields ##
        ## note, since we arent pre-loading the existing CSV with data before 1999, this number ##
        ## is an approximation ##
        ## since we will eventually throw out data pre-2022, this is fine (probably) ##
        df['start_number'] = df.groupby(['player_id']).cumcount() + 1
        return df[[
            'game_id', 'season', 'week', 'gameday', 'team', 'opponent', 'player_id', 'player_name', 'player_display_name',
            'start_number', 'rookie_year', 'entry_year', 'draft_number',
            'completions', 'attempts', 'passing_yards', 'passing_tds',
            'interceptions', 'sacks', 'carries', 'rushing_yards', 'rushing_tds'
        ]].copy()

    def calculate_raw_value(self, df):
        ## takes a df, with properly named fields and returns a series w/ VALUE ##
        ## formula for reference ##
        ## https://fivethirtyeight.com/methodology/how-our-nfl-predictions-work/ ##
        ##      -2.2 * Pass Attempts +
        ##         3.7 * Completions +
        ##       (Passing Yards / 5) +
        ##        11.3 * Passing TDs –
        ##      14.1 * Interceptions –
        ##          8 * Times Sacked –
        ##       1.1 * Rush Attempts +
        ##       0.6 * Rushing Yards +
        ##        15.9 * Rushing TDs
        return (
                -2.2 * df['attempts'] +
                3.7 * df['completions'] +
                (df['passing_yards'] / 5) +
                11.3 * df['passing_tds'] -
                14.1 * df['interceptions'] -
                8 * df['sacks'] -
                1.1 * df['carries'] +
                0.6 * df['rushing_yards'] +
                15.9 * df['rushing_tds']
        )

    def pull_data(self):
        ## wrapper for all the above functions ##
        print('Retrieving nflverse data...')
        df = self.retrieve_player_stats()
        while df is not None:
            ## data retrieval ##
            df = self.retrieve_player_meta(df)
            df = self.add_game_data(df)
            ## merge and format ##
            ## team stats ##
            df_team = self.aggregate_team_stats(df)
            df_team['team_VALUE'] = self.calculate_raw_value(df_team)
            df_team = df_team.drop(columns=self.stat_cols)
            ## df ##
            df = self.iso_top_passer(df)
            df = self.format_top_passer(df)
            df['player_VALUE'] = self.calculate_raw_value(df)
            df = df.drop(columns=self.stat_cols)
            ## create model file ##
            df = pd.merge(
                df,
                df_team[['game_id', 'team', 'team_VALUE']],
                on=['game_id', 'team'],
                how='left'
            )
            df = pd.merge(left=df, right=df_pff_ids, how="left", left_on="player_display_name", right_on='Player').drop(
                columns=["Player", "Pos", "Team"])

            df["player_id"] = df["PFF Ref"].copy()

            self.model_df = df.copy()
            print('     Successfully retrieved and stored')
            ## end loop ##
            df = None


class EloConstructor():
    ## this class takes in the updated model data, next weeks games, and ##
    ## the original elo file to create a new elo file in the same format ##
    def __init__(self, games, qb_model, at_wrapper,export_loc):
        self.games = games.copy()
        self.qb_model = qb_model  ## an updated QBModel Class object ##
        self.at_wrapper = at_wrapper ## an updated AirtableWrapper Class object ##
        self.export_loc = export_loc  ## location to export new file ##
        self.qb_values = pd.DataFrame(qb_model.data)
        self.original_elo_file = pd.read_csv(self.qb_model.original_file_loc, index_col=0)  ## original elo file ##
        self.original_elo_cols = self.original_elo_file.columns.to_list()
        self.new_games = None  ## games that occured after original file ##
        self.next_games = None  ## next weeks games ##
        self.new_file_games = None  ## merged new games and next games ##
        self.new_file_data = []  ## formatted rows to be appended to the existing ##
        self.new_file = None

    def determine_new_games(self):
        ## could do dynamically, but just assume anything after 2023-02-12 is new ##
        ## this df represents all played games since original elo file ##
        ## was last updated ##
        self.new_games = self.games[
            (self.games['gameday'] > '2023-02-12') &
            (~pd.isnull(self.games['result']))
            ].copy()
        if len(self.new_games) == 0:
            self.new_games = None

    def add_qbs_to_new_games(self):
        ## combine model_df, which is flat, with new games ##
        ## elo file is not flat ##
        ## if new games is none, update ##
        if self.new_games is None:
            self.determine_new_games()
        ## if there have been no new games, return without updating ##
        if self.new_games is None:
            return
        ## add home qb ##





        self.new_games = pd.merge(
            self.new_games,
            self.qb_values[[
                'game_id', 'team', 'player_id', 'player_display_name',
                'qb_value_pre', 'qb_adj', 'player_VALUE_adj', 'qb_value_post'
            ]].rename(columns={
                'team': 'home_team',
                'player_id': 'qb1_id',
                'player_display_name': 'qb1',
                'qb_value_pre': 'qb1_value_pre',
                'qb_adj': 'qb1_adj',
                'player_VALUE_adj': 'qb1_game_value',
                'qb_value_post': 'qb1_value_post'
            }),
            on=['game_id', 'home_team'],
            how='left'
        )
        print(self.qb_values.columns)
        ## add away qb ##
        self.new_games = pd.merge(
            self.new_games,
            self.qb_values[[
                'game_id', 'team', 'player_id', 'player_display_name',
                'qb_value_pre', 'qb_adj', 'player_VALUE_adj', 'qb_value_post'
            ]].rename(columns={
                'team': 'away_team',
                'player_id': 'qb2_id',
                'player_display_name': 'qb2',
                'qb_value_pre': 'qb2_value_pre',
                'qb_adj': 'qb2_adj',
                'player_VALUE_adj': 'qb2_game_value',
                'qb_value_post': 'qb2_value_post'
            }),
            on=['game_id', 'away_team'],
            how='left'
        )

    # def get_next_games(self):
    #     ## determine the next week of games ##
    #     unplayed = self.games[
    #         (pd.isnull(self.games['result']))
    #     ].copy()
    #     ## if there are no games, b/c the season is over, stop ##
    #     if len(unplayed) == 0:
    #         return None
    #     ## if there is a next week, filter games ##
    #     self.next_games = self.games[
    #         (self.games['season'] == unplayed.iloc[0]['season']) &
    #         (self.games['week'] == unplayed.iloc[0]['week'])
    #         ].copy()

    def get_next_games(self):
        ## determine the next week of games ##
        unplayed = self.games[
            (pd.isnull(self.games['result']))
        ].copy()

        ## if there are no games, b/c the season is over, stop ##
        if len(unplayed) == 0:
            return None

        ## filter unplayed games for the current week ##
        current_week_unplayed = unplayed[
            (unplayed['season'] == unplayed.iloc[0]['season']) &
            (unplayed['week'] == unplayed.iloc[0]['week'])
            ].copy()

        ## if there's only one unplayed game left in the current week ##
        if len(current_week_unplayed) <=2:
            next_week_games = self.games[
                (self.games['season'] == unplayed.iloc[0]['season']) &
                (self.games['week'] == unplayed.iloc[0]['week'] + 1) &
                (~self.games['home_team'].isin(
                    current_week_unplayed['home_team'].tolist() + current_week_unplayed['away_team'].tolist())) &
                (~self.games['away_team'].isin(
                    current_week_unplayed['home_team'].tolist() + current_week_unplayed['away_team'].tolist()))
                ].copy()

            self.next_games = pd.concat([current_week_unplayed, next_week_games])
        else:
            self.next_games = current_week_unplayed

    def extract_starter_values(self, qb_id, season, team, draft_number, gameday):
        ## helper function that pulls starters current value from the model ##
        ## and does necessary regression ##
        ## first create a random id if qb_id is null, which can happen if QB is not in roster ##
        ## file yet ##
        if pd.isnull(qb_id):
            qb_id = '00-' + str(numpy.random.randint(100000, 200000))
        ## create a dummy 'row' so we can use get_qb_value function from model ##
        row = {
            'player_id': qb_id,
            'season': season,
            'team': team,
            'draft_number': draft_number,
            'gameday': gameday
        }
        ## get the qb value ##
        qb_value = self.qb_model.get_qb_value(row)
        ## return the value ##
        return qb_value

    def add_starters(self):
        ## add starters and values to the new games ##
        ## this also needs to handle season over season regressions ##
        ## update starters ##
        self.at_wrapper.pull_current_starters()
        ## convert starters to a dict ##
        starter_dict = {}

        print(f"Starters from Air Table:{self.at_wrapper.starters_df}")

        for index, row in self.at_wrapper.starters_df.iterrows():

            player_display_name = row['player_display_name']
            player_display_name = player_display_name[0] if isinstance(player_display_name, list) else player_display_name
            draft_number = row['draft_number']
            draft_number = draft_number[0] if isinstance(draft_number, list) else draft_number

            starter_dict[row['team']] = {}
            starter_dict[row['team']]['qb_id'] = row['player_id']
            starter_dict[row['team']]['qb_name'] = player_display_name
            starter_dict[row['team']]['draft_number'] = draft_number

        ## helper func to apply to new games ##
        def apply_starters(row, starter_dict):
            ## home ##
            row['qb1_id'] = starter_dict[row['home_team']]['qb_id']
            row['qb1'] = starter_dict[row['home_team']]['qb_name']
            row['qb1_value_pre'] = self.extract_starter_values(
                row['qb1_id'], row['season'], row['home_team'],
                starter_dict[row['home_team']]['draft_number'], row['gameday']
            )
            row['qb1_game_value'] = numpy.nan
            row['qb1_value_post'] = numpy.nan
            ## away ##
            row['qb2_id'] = starter_dict[row['away_team']]['qb_id']
            row['qb2'] = starter_dict[row['away_team']]['qb_name']
            row['qb2_value_pre'] = self.extract_starter_values(
                row['qb2_id'], row['season'], row['away_team'],
                starter_dict[row['away_team']]['draft_number'], row['gameday']
            )
            row['qb2_game_value'] = numpy.nan
            row['qb2_value_post'] = numpy.nan
            ## return ##
            return row

        ## apply ##

        self.next_games.to_csv('D:/NFL/QB2023PreSeason/next_games.csv')

        self.next_games = self.next_games.apply(
            apply_starters,
            starter_dict=starter_dict,
            axis=1
        )

    def add_team_values(self):
        ## once next week has been updated with starter values, add team values ##
        ## and make adjustments ##
        ## first set the models current week to the week of next games ##
        self.qb_model.current_week = self.next_games.iloc[-1]['week']

        ## helper to add team values ##
        def apply_team_values(row):
            ## home ##
            home_val, home_adj = self.qb_model.get_team_off_value(
                row['home_team'], row['qb1_value_pre'], row['season']
            )
            ## away ##
            away_val, away_adj = self.qb_model.get_team_off_value(
                row['away_team'], row['qb2_value_pre'], row['season']
            )
            ## add adjs to row ##
            row['qb1_adj'] = home_adj
            row['qb2_adj'] = away_adj
            ## return ##
            return row

        ## apply ##
        self.next_games = self.next_games.apply(
            apply_team_values,
            axis=1
        )

    def merge_new_and_next(self):
        ## merge new games and next games with logic to handle blanks ##
        if self.new_games is None:
            if self.next_games is None:
                ## if both are none, return none ##
                return
            else:
                self.new_file_games = self.next_games
        else:
            if self.next_games is None:
                self.new_file_games = self.new_games
            else:
                ## merge ##
                ## align columns ##
                self.new_file_games = pd.concat([
                    self.new_games,
                    self.next_games[
                        self.new_games.columns
                    ]
                ])

    def format_games_row(self, row):
        ## takes a row from model_df and formats it for the elo file ##
        new_row = {}
        for col in self.original_elo_cols:
            new_row[col] = numpy.nan
        ## add in the values ##
        new_row['date'] = row['gameday']
        new_row['season'] = row['season']
        new_row['team1'] = row['home_team']
        new_row['team2'] = row['away_team']
        new_row['score1'] = row['home_score']
        new_row['score2'] = row['away_score']
        ## qb values ##
        ## each qb value is in VALUE, but needs to be in elo, so multiply by 3.3 ##
        new_row['qb1'] = row['qb1']
        new_row['qb2'] = row['qb2']
        new_row['qb1_value_pre'] = row['qb1_value_pre'] * 3.3
        new_row['qb2_value_pre'] = row['qb2_value_pre'] * 3.3
        new_row['qb1_value_post'] = row['qb1_value_post'] * 3.3
        new_row['qb2_value_post'] = row['qb2_value_post'] * 3.3
        new_row['qb1_adj'] = row['qb1_adj'] * 3.3
        new_row['qb2_adj'] = row['qb2_adj'] * 3.3
        new_row['qb1_game_value'] = row['qb1_game_value'] * 3.3
        new_row['qb2_game_value'] = row['qb2_game_value'] * 3.3
        ## netural locs ##
        if row['location'] == 'Home':
            new_row['neutral'] = 0
        else:
            new_row['neutral'] = 1
        ## playoffs ##
        if row['game_type'] == 'REG':
            new_row['playoff'] = numpy.nan
        else:
            new_row['playoff'] = 1
        ## write row to new file data ##
        self.new_file_data.append(new_row)

    def create_new_file(self):
        ## merges original elo file with new games and next games and then saves ##
        ## to the root of the package ##
        if self.new_file_games is None:
            self.new_file = self.original_elo_file
        else:
            ## sort games ##
            self.new_file_games = self.new_file_games.sort_values(
                by=['season', 'week', 'gameday'],
                ascending=[True, True, True]
            ).reset_index(drop=True)
            ## parse rows ##
            for index, row in self.new_file_games.iterrows():
                self.format_games_row(row)
            ## create new df and concat ##
            self.new_file = pd.concat([
                self.original_elo_file,
                pd.DataFrame(self.new_file_data)
            ])
            self.new_file = self.new_file.reset_index(drop=True)

    def construct_elo_file(self):
        ## wrapper on the above functions that creates the elo file ##
        print('Constructing elo file...')
        print('     Determining new games...')
        self.determine_new_games()
        if self.new_games is None:
            print('     No new games found')
        else:
            print('          Found {0} new games. Adding QB data...'.format(len(self.new_games)))
            self.add_qbs_to_new_games()
        print('     Determining next games...')
        self.get_next_games()
        if self.next_games is None:
            print('     No next games found')
        else:
            print('          Found {0} next games. Pulling projected starters...'.format(len(self.next_games)))
            self.add_starters()
            print('          Adding team values for adjustments...')
            self.add_team_values()

        print('     Merging new and next games...')
        self.merge_new_and_next()
        if self.new_file_games is None:
            print('     No new games to merge to original elo file. Will not update')
        else:
            print('     Formatting games for elo file...')
            self.create_new_file()
            print('     Saving new elo file...')
            self.new_file.to_csv(
                '{0}/qb_elos.csv'.format(self.export_loc),
                index=False
            )
            print('     Done')


def run(perform_starter_update=True, model_only=False):
    ## load configs and secrets ##
    config = None
    secrets = None
    package_folder = "D:/NFL/QB2023PreSeason"
    with open('{0}/model_config.json'.format(package_folder)) as fp:
        config = json.load(fp)
    with open('{0}/secrets.json'.format(package_folder)) as fp:
        secrets = json.load(fp)
    ## load data ##
    data = DataLoader()
    ## run model ##
    print('Running model...')
    data.model_df.to_csv('D:/NFL/QB2023PreSeason/model_df.csv', index=False)
    model = QBModel(data.model_df, config)
    data.games.to_csv('D:/NFL/QB2023PreSeason/games.csv', index=False)
    model.run_model()
    if model_only:
        return model
    ## update starters ##
    at_wrapper = AirtableWrapper(
        model.games,
        secrets['airtable'],
        perform_starter_update=perform_starter_update
    )
    at_wrapper.update_qb_table()
    # at_wrapper.update_qb_options()
    at_wrapper.update_starters()
    at_wrapper.update_qb_table_games_started()
    ## pause and wait for confirmation that manual edits have been made in airtable ##
    decision = input('When starters have been updated in Airtable, type "RUN" and press enter:')
    print(decision)
    ## construct elo file ##

    if decision == 'RUN':

        constructor = EloConstructor(
            data.games,
            model,
            at_wrapper,
            package_folder
        )
        constructor.construct_elo_file()

    starters_exist = at_wrapper.existing_starters

    with open('D:/NFL/NFL Beyond/current_starters.pkl', 'wb') as f:
        pickle.dump(at_wrapper.existing_starters, f)


run(perform_starter_update=True, model_only=False)
