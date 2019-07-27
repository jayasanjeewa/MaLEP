--https://www.spec.org/power_ssj2008/results/power_ssj2008.html

CREATE TABLE server_specs(
   id        VARCHAR(10) NOT NULL PRIMARY KEY
  ,vendor    VARCHAR(200)
  ,system    VARCHAR(400)
  ,nodes     int(11) DEFAULT NULL
  ,jvm_vendor VARCHAR(200)
  ,cpu_desc  VARCHAR(28)
  ,cpu_speed int(11) DEFAULT NULL
  ,chips    int(11) DEFAULT NULL
  ,cores     int(11) DEFAULT NULL
  ,threads   int(11) DEFAULT NULL
  ,memory_gb    int(11) DEFAULT NULL
  ,ssj_ops_full   int(11) DEFAULT NULL
  ,avg_watts_full_util int(11) DEFAULT NULL
  ,avg_watts_idle int(11) DEFAULT NULL
  ,overall_ssj_ops int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

INSERT INTO server_specs(id, vendor, system, nodes, jvm_vendor, cpu_desc, cpu_speed, chips, cores, threads, memory_gb, ssj_ops_full, avg_watts_full_util, avg_watts_idle, overall_ssj_ops)
VALUES                  (1, 'ASUSTeK Computer Inc.', 'RS720-E9/RS', 1, 'Oracle Corporation' , 'Intel Xeon Platinum 8180' ,2500, 2, 56, 112, 192, 5386401, 385, 48.2, 12727);
