#include <QApplication>
#include <QMainWindow>
#include <QWidget>
#include <QHBoxLayout>
#include <QSlider>
#include <Qt>
#include <QLabel>
#include <QVBoxLayout>
#include <QScrollArea>
#include <QVector>
#include <QPushButton>
#include <QTimer>
#include <QFrame>

#include "openglwidget.h"
#include "mhridentitydata.h"
#include "hoverable_slider.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    MHRIdentityData mhrData;

    if (!mhrData.load("../../models/mhr_identity.bin"))
    {
        qDebug() << "MHR data loading failed.";
    }

    QMainWindow window;
    window.setWindowTitle("Avatar Studio");
    window.resize(1200, 800);

    QWidget* centralWidget = new QWidget(&window);    
    QHBoxLayout* layout = new QHBoxLayout(centralWidget);

    QWidget* controlPanel = new QWidget(centralWidget);
    QVBoxLayout* controlLayout = new QVBoxLayout(controlPanel);
    controlPanel->setMinimumWidth(300); // make control tabs wider

    QTabWidget *tabWidget = new QTabWidget(controlPanel); // template for parameters tabs
    controlLayout->addWidget(tabWidget);

    QPushButton *resetButton = new QPushButton("Reset", tabWidget);
    controlLayout->addWidget(resetButton);

    OpenGLWidget* openGLWidget = new OpenGLWidget(centralWidget);

    // Heatmap legend
    QWidget* legendWidget = new QWidget(centralWidget);
    QVBoxLayout* legendLayout = new QVBoxLayout(legendWidget);

    QLabel* legendTitle = new QLabel("Influence", legendWidget);
    legendTitle->setAlignment(Qt::AlignCenter);
    legendLayout->addWidget(legendTitle);

    QLabel* highLabel = new QLabel("High", legendWidget);
    highLabel->setAlignment(Qt::AlignCenter);
    legendLayout->addWidget(highLabel);

    QFrame* colorBar = new QFrame(legendWidget);
    colorBar->setFixedSize(24, 180);
    colorBar->setStyleSheet(
        "QFrame {"
        "border: 1px solid #777;"
        "border-radius: 3px;"
        "background: qlineargradient(x1:0, y1:0, x2:0, y2:1,"
        "stop:0 red,"
        "stop:0.25 yellow,"
        "stop:0.5 green,"
        "stop:0.75 cyan,"
        "stop:1 blue);"
        "}"
        );

    legendLayout->addWidget(colorBar, 0, Qt::AlignCenter);

    QLabel* lowLabel = new QLabel("Low", legendWidget);
    lowLabel->setAlignment(Qt::AlignCenter);
    legendLayout->addWidget(lowLabel);

    legendLayout->addStretch();

    QWidget *bodyTab = new QWidget(); // body tab
    QWidget *headTab = new QWidget(); // head tab
    QWidget *handsTab = new QWidget(); // hands tab
    tabWidget->addTab(bodyTab, "Body");
    tabWidget->addTab(headTab, "Head");
    tabWidget->addTab(handsTab, "Hands");

    int parameterCount = openGLWidget->identityParameterCount();

    // Body tab
    QVBoxLayout *bodyTabLayout = new QVBoxLayout(bodyTab);

    QScrollArea *bodyScrollArea = new QScrollArea(bodyTab);
    bodyScrollArea->setWidgetResizable(true);

    bodyTabLayout->addWidget(bodyScrollArea);

    QWidget *bodyParametersWidget = new QWidget(bodyScrollArea);
    QVBoxLayout *bodyParametersLayout =
        new QVBoxLayout(bodyParametersWidget);
    bodyScrollArea->setWidget(bodyParametersWidget);

    // Head tab
    QVBoxLayout *headTabLayout = new QVBoxLayout(headTab);

    QScrollArea *headScrollArea = new QScrollArea(headTab);
    headScrollArea->setWidgetResizable(true);

    headTabLayout->addWidget(headScrollArea);

    QWidget *headParametersWidget = new QWidget();
    QVBoxLayout *headParametersLayout =
        new QVBoxLayout(headParametersWidget);

    headScrollArea->setWidget(headParametersWidget);

    // Hands tab
    QVBoxLayout *handsTabLayout = new QVBoxLayout(handsTab);

    QScrollArea *handsScrollArea = new QScrollArea(handsTab);
    handsScrollArea->setWidgetResizable(true);

    handsTabLayout->addWidget(handsScrollArea);

    QWidget *handsParametersWidget = new QWidget(); // where hands the parameteres live
    QVBoxLayout *handsParametersLayout =
        new QVBoxLayout(handsParametersWidget);

    handsScrollArea->setWidget(handsParametersWidget);

    QVector<QSlider*> sliders;

    for (int i = 0; i < parameterCount; ++i)
    {
        // Introduce 3 groups of parameters
        QVBoxLayout *targetLayout = nullptr;

        if (i < 20)
        {
            targetLayout = bodyParametersLayout;
        }
        else if (i < 40)
        {
            targetLayout = headParametersLayout;
        }
        else
        {
            targetLayout = handsParametersLayout;
        }

        HoverableSlider* slider = new HoverableSlider(Qt::Horizontal);

        const auto& parameter = openGLWidget->identityParameter(i);

        slider->setRange(
            parameter.min * 100,
            parameter.max * 100
            );

        slider->setValue(0);
        sliders.append(slider); // store sliders in collection

        // detec hover
        QObject::connect(
            slider, &HoverableSlider::hovered,
            [openGLWidget, i]()
            {
                openGLWidget->setHeatmapParameter(i);
            }
        );

        QObject::connect(
            slider, &HoverableSlider::unhovered,
            [openGLWidget, slider]()
            {
                QTimer::singleShot(0, openGLWidget, [openGLWidget, slider]()
                {
                    if (!slider->underMouse())
                        openGLWidget->clearHeatmap();
                });
            }
        );

        QLabel* label = new QLabel(
            QString("%1: %2")
                .arg(parameter.name)
                .arg(parameter.value, 0, 'f', 2)
            );

        QString parameterName = parameter.name;

        QObject::connect(slider, &QSlider::valueChanged,
                         [openGLWidget, label, i, parameterName](int value)
                         {
                             float parameterValue = value / 100.0f;

                             openGLWidget->setIdentityParameter(i, parameterValue);

                             label->setText(
                                 QString("%1: %2")
                                     .arg(parameterName)
                                     .arg(parameterValue, 0, 'f', 2)
                                 );
                         });

        QVBoxLayout* singleParameterLayout = new QVBoxLayout();

        singleParameterLayout->addWidget(label);
        singleParameterLayout->addWidget(slider);

        targetLayout->addLayout(singleParameterLayout);
    }

    QObject::connect(resetButton, &QPushButton::clicked,
                     bodyParametersWidget,
                     [sliders]()
                     {
                         for (auto* slider : sliders) {
                             slider->setValue(0);
                         }
                     });

    layout->addWidget(openGLWidget, 1);
    layout->addWidget(legendWidget);
    layout->addWidget(controlPanel);

    window.setCentralWidget(centralWidget);
    window.show();

    return app.exec();
}