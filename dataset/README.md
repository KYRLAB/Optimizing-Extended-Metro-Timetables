## Datasets

We collect three kinds of data: metro ridership counts for Seoul and Busan, weekday train schedules from 00:00 to 02:00, and extended metro timetables.

### Download datasets

- The Seoul metro ridership data provided by the [Seoul Open Data Plaza](https://data.seoul.go.kr/dataList/OA-12921/F/1/datasetView.do). 
- The Busan metro ridership data provided by the [Open Government Data portal](https://www.data.go.kr/data/3057229/fileData.do). 
- The Seoul and Busan metro timetables provided by the [SEOUL METRO Official Website](http://www.seoulmetro.co.kr/en) and [Busan Transportation Coporation Official Website](https://www.humetro.busan.kr), respectively.

### Sample datasets

- (Sample Data) Seoul/Busan Metro Timetable: This is a sample of the metro operating timetable for the period from Dec 31, 2023 to Jan 1, 2024. Cells highlighted in yellow represent trains that were added as extra services during the year-end.
- (Sample Data) Seoul/Busan Metro Passenger Boarding and Alighting: This is a sample of metro boarding and alighting data for the period from Dec 31, 2023 to Jan 1, 2024.

### Folder structure

```
├── dataset
│   ├── seoul_line_1.csv
│   ├── busan_line_1.csv
│   ├── *.csv (Another dataset following the same schema)
└── └── README.md
```

---

