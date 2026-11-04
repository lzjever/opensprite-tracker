"""All-body scheduling contract and conservative physical wire budget."""
import csv

import pytest

from opensprite_sdk.backends.hardware_config import HardwareConfig
from test_hardware_config import measured_fixture


def test_every_motor_generated_at500hz_and_four_buses_fit_budget(tmp_path):
    config=HardwareConfig.from_dict(measured_fixture())
    files=config.generate(tmp_path)
    with open(files['motors']) as stream:
        rows=list(csv.DictReader(stream,delimiter='\t'))
    assert len(rows)==31 and {row['poll_rate_hz'] for row in rows}=={'500'}
    budgets=config.can_wire_budget()
    assert len(budgets)==4 and max(b['motors'] for b in budgets)==8
    assert max(b['utilization'] for b in budgets)==pytest.approx(.7168)
    assert all(not b['hardware_measured'] for b in budgets)


@pytest.mark.parametrize('arbitration,data',[(1_000_000,2_000_000),(500_000,5_000_000)])
def test_insufficient_bus_budget_is_rejected_before_writing_native_config(tmp_path,arbitration,data):
    config=measured_fixture()
    config['can_adapter'].update(arbitration_bitrate=arbitration,data_bitrate=data)
    with pytest.raises(ValueError,match='budget'):
        HardwareConfig.from_dict(config).generate(tmp_path)
    assert not (tmp_path/'motors.tsv').exists()
