import requests
import time
import json
import pandas as pd
import numpy as np

class AirtableWrapper:
    def __init__(self, model_df, at_config, perform_starter_update=True):
        self.model_df = model_df
        self.at_config = at_config
        self.perform_starter_update = perform_starter_update

        self.base = at_config['base']
        self.qb_table = at_config['qb_table']
        self.starter_table = at_config['starter_table']
        self.token = at_config['token']
        self.dropdown_field_id = at_config['dropdown_field']

        self.base_headers = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        }

        self.existing_qbs = []
        self.qb_options = []
        self.existing_starters = {}
        self.starters_df = None

    def _get(self, table, params={}):
        url = f'https://api.airtable.com/v0/{self.base}/{table}'
        time.sleep(0.25)
        return requests.get(url, headers=self.base_headers, params=params)

    def _patch(self, table, data):
        url = f'https://api.airtable.com/v0/{self.base}/{table}'
        time.sleep(0.25)
        return requests.patch(url, headers=self.base_headers, data=json.dumps(data))

    def _post(self, table, data):
        url = f'https://api.airtable.com/v0/{self.base}/{table}'
        time.sleep(0.25)
        return requests.post(url, headers=self.base_headers, data=json.dumps(data))

    def _get_paginated(self, table, params={}):
        offset = None
        all_records = []
        while True:
            if offset:
                params['offset'] = offset
            resp = self._get(table, params)
            result = resp.json()
            all_records.extend(result.get('records', []))
            offset = result.get('offset')
            if not offset:
                break
        return all_records

    def get_existing_qbs(self):
        records = self._get_paginated(self.qb_table)
        self.existing_qbs = [r['fields']['player_id'] for r in records if 'player_id' in r['fields']]

    def get_qb_options(self):
        url = f'https://api.airtable.com/v0/meta/bases/{self.base}/tables'
        resp = requests.get(url, headers=self.base_headers).json()
        for table in resp['tables']:
            if table['id'] == self.starter_table:
                for field in table['fields']:
                    if field['id'] == self.dropdown_field_id:
                        self.qb_options = [opt['name'] for opt in field['options']['choices']]

    def get_starters(self):
        records = self._get_paginated(self.starter_table)
        for r in records:
            f = r.get('fields', {})
            if 'team' in f and 'player_id' in f:
                self.existing_starters[f['team']] = {
                    'record_id': r['id'],
                    'qb_id': f['player_id'][0] if isinstance(f['player_id'], list) else f['player_id']
                }

    def write_qbs(self, df):
        records = []
        for _, row in df.iterrows():
            fields = {k: (None if pd.isnull(v) else v) for k, v in row.items()}
            records.append({"fields": fields})
        data = {"records": records, "typecast": True}
        self._post(self.qb_table, data)

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

            draft_number = record['fields']['draft_pick'][0] if isinstance(record['fields']['draft_pick'], list) else record['fields']['draft_pick']

            start_number = record['fields']['start_number'][0] if isinstance(record['fields']['start_number'], list) else record['fields']['start_number']

            starters_data.append({
                'team': record['fields']['team'],
                'player_id': player_id,
                'player_display_name': player_display_name,
                'draft_pick': draft_number,
                'start_number': start_number
            })
        ## write ##
        self.starters_df = pd.DataFrame(starters_data)

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

    def update_starters(self):
        if not self.perform_starter_update:
            return

        self.get_starters()
        last = self.model_df.sort_values("gameday", ascending=False).drop_duplicates("team")
        last['qb_id'] = last['player_display_name'] + '-' + last['player_id'].astype(str)

        writes, updates = [], []

        for _, row in last.iterrows():
            team = row['team']
            qb_id = row['qb_id']
            if team in self.existing_starters:
                if self.existing_starters[team]['qb_id'] != qb_id:
                    updates.append({
                        'id': self.existing_starters[team]['record_id'],
                        'fields': {'qb_id': qb_id}
                    })
            else:
                writes.append({"fields": {'team': team, 'qb_id': qb_id}})

        if writes:
            self._post(self.starter_table, {"records": writes, "typecast": True})
        if updates:
            self._patch(self.starter_table, {"records": updates, "typecast": True})
