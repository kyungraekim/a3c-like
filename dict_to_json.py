import json

DUMP_OPTIONS = {
    'sort_keys': True,
    'indent': 4,
}


def generate_actors():
    d = []
    for i in range(5):
        d.append({
            'host': 'localhost',
            'port': 38080 + i,
            'total_steps': 10,
            'env': 'sls',
        })
    with open('actor-list.json', 'w') as f_out:
        f_out.write(json.dumps(d, **DUMP_OPTIONS))


def generate_learner():
    with open('learner.json', 'w') as f_out:
        f_out.write(json.dumps({
            'total_steps': 10,
            'env_name': 'sls',
        }, **DUMP_OPTIONS))


if __name__ == '__main__':
    generate_actors()
    generate_learner()
