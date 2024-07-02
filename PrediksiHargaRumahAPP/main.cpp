#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <string>
#include <Windows.h>
#include <commctrl.h>
#include <cstdio>
#include <cwchar>
#include <fstream>
#include <filesystem>
#include <cstdlib>
#include <locale>
#include <codecvt>

#pragma comment(lib, "comctl32.lib")

HWND g_hButton;
HWND g_hComboBox;
HWND g_hComboBox0;
HWND g_hComboBox1;
HWND g_hComboBox2;
HWND g_hListView;
HWND hwnd;

class TextBox{
private:
    HINSTANCE hInstance;
    HWND parentWnd;
    int xPos, yPos, boxWidth, boxHeight;
public:
    HWND g_hTextBox;
    TextBox(HWND parentWnd, HINSTANCE hInstance, int x, int y, int width, int height)
        : hInstance(hInstance), g_hTextBox(NULL), parentWnd(parentWnd), xPos(x), yPos(y), boxWidth(width), boxHeight(height) {}

    TextBox()
        : hInstance(NULL), g_hTextBox(NULL), parentWnd(NULL), xPos(NULL), yPos(NULL), boxWidth(NULL), boxHeight(NULL) {}

    void Create() {
        std::cout << "creating" << std::endl;
        g_hTextBox = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"",
            WS_CHILD | WS_VISIBLE | ES_AUTOHSCROLL,
            xPos, yPos, boxWidth, boxHeight,
            parentWnd, NULL, hInstance, NULL);

    }

    void CreateLabel(const wchar_t label[64]) {
        std::cout << "ADSADS" << std::endl;
        g_hTextBox = CreateWindowExW(
            0, L"STATIC", label,
            WS_VISIBLE | WS_CHILD | SS_LEFT,
            xPos, yPos, boxWidth, boxHeight,
            parentWnd, NULL, hInstance, NULL);
    }

    void GetText(wchar_t* buffer, int bufferSize) const {
        if (g_hTextBox) {
            if (g_hTextBox) {
                GetWindowText(g_hTextBox, buffer, bufferSize);
            }
        }
    }

    void Setter(HWND parentWnd, HINSTANCE hInstance, int x, int y, int width, int height) {
        this->parentWnd = parentWnd;
        this->hInstance = hInstance;
        this->xPos = x;
        this->yPos = y;
        this->boxWidth = width;
        this->boxHeight = height;
    }

};


TextBox g_hTextBox0;
TextBox g_hLabel0;
TextBox g_hTextBox1;
TextBox g_hLabel1;
TextBox g_hTextBox2;
TextBox g_hLabel2;
TextBox g_hTextBox3;
TextBox g_hLabel3;
TextBox g_hTextBox4;
TextBox g_hLabel4;
TextBox g_hTextBox5;
TextBox g_hLabel5;
TextBox g_hLabel6;

TextBox g_hTextBox7;
TextBox g_hLabel7;
TextBox g_hLabel8;
TextBox g_hLabel9;

TextBox g_hTextBox10;
TextBox g_hLabel10;
TextBox g_hTextBox11;
TextBox g_hLabel11;
TextBox g_hTextBox12;
TextBox g_hLabel12;
TextBox g_hLabel13;

HWND hasil;
wchar_t isiHasil[64] = L"Hasil: ";

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

LPCTSTR szWindowClass = TEXT("GraphWindowClass");
LPCTSTR szTitle = TEXT("Graph using Windows API");

const wchar_t arr[14][39] = { L"harga rumah", L"jumlah kamar tidur", L"jumlah kamar mandi",
                L"luas tanah (m2)", L"luas bangunan (m2)", L"carport (mobil)",
                L"pasokan listrik (watt)", L"Kabupaten/Kota", L"kecamatan",
                L"keamanan (ada/tidak)", L"taman (ada/tidak)",
                L"jarak dengan rumah sakit terdekat (km)",
                L"jarak dengan sekolah terdekat (km)", L"jarak dengan tol terdekat (km)" };


void RedirectIOToConsole();

// Entry point for the application
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{


    RedirectIOToConsole();

    INITCOMMONCONTROLSEX icex;
    icex.dwSize = sizeof(INITCOMMONCONTROLSEX);
    icex.dwICC = ICC_LISTVIEW_CLASSES; // Include list-view control class
    InitCommonControlsEx(&icex);

    // Define and register the window class
    const wchar_t CLASS_NAME[] = L"SampleWindowClass";

    WNDCLASS wc = {};

    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;

    RegisterClass(&wc);

    // Create the window
    hwnd = CreateWindowExW(
        0,                              // Optional window styles.
        CLASS_NAME,                     // Window class
        L"Learn to Program Windows",     // Window text
        //WS_OVERLAPPEDWINDOW | WS_VISIBLE,            // Window style
        WS_OVERLAPPEDWINDOW | WS_VSCROLL | WS_HSCROLL,
        // Size and position
        0, 0, GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN),

        NULL,       // Parent window    
        NULL,       // Menu
        hInstance,  // Instance handle
        NULL        // Additional application data
    );

    if (hwnd == NULL)
    {
        return 0;
    }
    


    ShowWindow(hwnd, SW_MAXIMIZE);
    UpdateWindow(hwnd);

    MSG msg = {};
    while (GetMessage(&msg, NULL, 0, 0))
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    FreeConsole();


    return msg.wParam;
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    switch (uMsg)
    {
        case WM_CREATE:
        {
            g_hTextBox0.Setter(hwnd, NULL, 50, 50, 250, 25);
            g_hLabel0.Setter(hwnd, NULL, 50, 25, 250, 25);
            g_hTextBox1.Setter(hwnd, NULL, 50, 100, 250, 25);
            g_hLabel1.Setter(hwnd, NULL, 50, 75, 250, 25);
            g_hTextBox2.Setter(hwnd, NULL, 50, 150, 250, 25);
            g_hLabel2.Setter(hwnd, NULL, 50, 125, 250, 25);
            g_hTextBox3.Setter(hwnd, NULL, 50, 200, 250, 25);
            g_hLabel3.Setter(hwnd, NULL, 50, 175, 250, 25);
            g_hTextBox4.Setter(hwnd, NULL, 50, 250, 250, 25);
            g_hLabel4.Setter(hwnd, NULL, 50, 225, 250, 25);
            g_hTextBox5.Setter(hwnd, NULL, 50, 300, 250, 25);
            g_hLabel5.Setter(hwnd, NULL, 50, 275, 250, 25);

            g_hTextBox0.Create();
            g_hLabel0.CreateLabel(L"jumlah kamar tidur");

            g_hButton = CreateWindowExW(
                0, L"BUTTON",       // predefined class 
                L"FORECAST",        // button text 
                WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                350, 50, 100, 25,   // set size in WM_SIZE message 
                hwnd,               // parent window 
                NULL,               // No menu.
                NULL,          // instance 
                NULL                // no extra data 
            );


            g_hTextBox1.Create();
            g_hLabel1.CreateLabel(L"jumlah kamar mandi");

            g_hTextBox2.Create();
            g_hLabel2.CreateLabel(L"luas tanah (m2)");

            g_hTextBox3.Create();
            g_hLabel3.CreateLabel(L"luas bangunan (m2)");

            g_hTextBox4.Create();
            g_hLabel4.CreateLabel(L"carport (mobil)");

            g_hTextBox5.Create();
            g_hLabel5.CreateLabel(L"pasokan listrik (watt)");


            g_hComboBox = CreateWindowW(L"COMBOBOX", NULL, CBS_DROPDOWNLIST | CBS_HASSTRINGS | WS_CHILD | WS_VISIBLE | WS_VSCROLL,
                50, 350, 250, 200, hwnd, NULL, NULL, NULL);

            SendMessage(g_hComboBox, CB_ADDSTRING, 0, (LPARAM)L"Jakarta Pusat");
            SendMessage(g_hComboBox, CB_ADDSTRING, 0, (LPARAM)L"Jakarta Selatan");
            SendMessage(g_hComboBox, CB_ADDSTRING, 0, (LPARAM)L"Jakarta Timur");
            SendMessage(g_hComboBox, CB_ADDSTRING, 0, (LPARAM)L"Jakarta Utara");
            SendMessage(g_hComboBox, CB_ADDSTRING, 0, (LPARAM)L"Jakarta Barat");

            SendMessage(g_hComboBox, CB_SETCURSEL, 0, 0);

            g_hLabel6.Setter(hwnd, NULL, 50, 325, 250, 25);
            g_hLabel6.CreateLabel(L"Kabupaten/Kota");

            g_hLabel7.Setter(hwnd, NULL, 50, 375, 250, 25);
            g_hLabel7.CreateLabel(L"kecamatan");

            g_hTextBox7.Setter(hwnd, NULL, 50, 400, 250, 25);
            g_hTextBox7.Create();


            g_hLabel8.Setter(hwnd, NULL, 50, 425, 250, 25);
            g_hLabel8.CreateLabel(L"keamanan (ada/tidak)");

            g_hComboBox0 = CreateWindowW(L"COMBOBOX", NULL, CBS_DROPDOWNLIST | CBS_HASSTRINGS | WS_CHILD | WS_VISIBLE | WS_VSCROLL,
                50, 450, 250, 200, hwnd, NULL, NULL, NULL);

            SendMessage(g_hComboBox0, CB_ADDSTRING, 0, (LPARAM)L"ada");
            SendMessage(g_hComboBox0, CB_ADDSTRING, 0, (LPARAM)L"tidak");

            SendMessage(g_hComboBox0, CB_SETCURSEL, 0, 0);
            
            g_hLabel9.Setter(hwnd, NULL, 50, 475, 250, 25);
            g_hLabel9.CreateLabel(L"taman (ada/tidak)");

            g_hComboBox1 = CreateWindowW(L"COMBOBOX", NULL, CBS_DROPDOWNLIST | CBS_HASSTRINGS | WS_CHILD | WS_VISIBLE | WS_VSCROLL,
                50, 500, 250, 200, hwnd, NULL, NULL, NULL);

            SendMessage(g_hComboBox1, CB_ADDSTRING, 0, (LPARAM)L"ada");
            SendMessage(g_hComboBox1, CB_ADDSTRING, 0, (LPARAM)L"tidak");

            SendMessage(g_hComboBox1, CB_SETCURSEL, 0, 0);

            g_hLabel10.Setter(hwnd, NULL, 50, 525, 275, 25);
            g_hLabel10.CreateLabel(L"jarak dengan rumah sakit terdekat (km)");

            g_hTextBox10.Setter(hwnd, NULL, 50, 550, 275, 25);
            g_hTextBox10.Create();

            g_hLabel11.Setter(hwnd, NULL, 50, 575, 275, 25);
            g_hLabel11.CreateLabel(L"jarak dengan sekolah terdekat (km)");

            g_hTextBox11.Setter(hwnd, NULL, 50, 600, 275, 25);
            g_hTextBox11.Create();

            g_hLabel12.Setter(hwnd, NULL, 50, 625, 275, 25);
            g_hLabel12.CreateLabel(L"jarak dengan tol terdekat (km)");

            g_hTextBox12.Setter(hwnd, NULL, 50, 650, 275, 25);
            g_hTextBox12.Create();


            hasil = CreateWindowExW(
                0, L"STATIC", isiHasil,
                WS_VISIBLE | WS_CHILD | SS_LEFT,
                350, 100, 150, 25,
                hwnd, NULL, NULL, NULL);

            g_hListView = CreateWindowExW(
                0,                      // Optional styles
                WC_LISTVIEW,            // List view control class
                L"",                    // Not used when creating a list view
                // WS_VISIBLE | WS_CHILD | LVS_REPORT | LVS_EDITLABELS, // Styles
                WS_CHILD | WS_VISIBLE | LVS_REPORT | LVS_SINGLESEL | LVS_SHOWSELALWAYS | WS_BORDER,
                500, 53, 800, 400,       // Position and size
                hwnd,                 // Parent window
                NULL,                   // No menu
                NULL,              // Instance handle
                NULL                    // No additional data
            );

            
            LVCOLUMNW lvColumn;
            lvColumn.mask = LVCF_TEXT | LVCF_WIDTH | LVCF_SUBITEM;

            for (int i = 0; i < 14; i++) {
                lvColumn.cx = 120;
                lvColumn.pszText = const_cast<LPWSTR>(arr[i]);
                lvColumn.iSubItem = i;
                ListView_InsertColumn(g_hListView, i, &lvColumn);
            }

            g_hComboBox2 = CreateWindowW(L"COMBOBOX", NULL, CBS_DROPDOWNLIST | CBS_HASSTRINGS | WS_CHILD | WS_VISIBLE | WS_VSCROLL,
                1000, 500, 250, 200, hwnd, NULL, NULL, NULL);

            SendMessage(g_hComboBox2, CB_ADDSTRING, 0, (LPARAM)L"DecisionTree_model_normal_0.12_2295.joblib");
            SendMessage(g_hComboBox2, CB_ADDSTRING, 0, (LPARAM)L"GradientBoosting_model_IQR_0.76_1856.joblib");

            SendMessage(g_hComboBox2, CB_SETCURSEL, 0, 0);

            break;
        }

        case WM_COMMAND:
            // Check the identifier of the control sending the command message
            if (LOWORD(wParam) == BN_CLICKED && (HWND)lParam == g_hButton) {
                std::wcout << isiHasil << std::endl;

                const int bufferSize = 64;
                wchar_t bufferText[13][bufferSize] = { { 0 } };
                g_hTextBox0.GetText(bufferText[0], bufferSize);

                g_hTextBox1.GetText(bufferText[1], bufferSize);

                g_hTextBox2.GetText(bufferText[2], bufferSize);

                g_hTextBox3.GetText(bufferText[3], bufferSize);

                g_hTextBox4.GetText(bufferText[4], bufferSize);

                g_hTextBox5.GetText(bufferText[5], bufferSize);


                int itemIndex = SendMessage(g_hComboBox, CB_GETCURSEL, 0, 0);
                int textLength = SendMessage(g_hComboBox, CB_GETLBTEXTLEN, itemIndex, 0);
                std::wcout << L"itemIndex: " << itemIndex << std::endl;
                SendMessage(g_hComboBox, CB_GETLBTEXT, itemIndex, (LPARAM)(LPCTSTR)bufferText[6]);


                g_hTextBox7.GetText(bufferText[7], bufferSize);

                itemIndex = SendMessage(g_hComboBox0, CB_GETCURSEL, 0, 0);
                textLength = SendMessage(g_hComboBox0, CB_GETLBTEXTLEN, itemIndex, 0);
                SendMessage(g_hComboBox0, CB_GETLBTEXT, itemIndex, (LPARAM)(LPCTSTR)bufferText[8]);

                itemIndex = SendMessage(g_hComboBox1, CB_GETCURSEL, 0, 0);
                textLength = SendMessage(g_hComboBox1, CB_GETLBTEXTLEN, itemIndex, 0);
                SendMessage(g_hComboBox0, CB_GETLBTEXT, itemIndex, (LPARAM)(LPCTSTR)bufferText[9]);
                std::cout << bufferText[9] << std::endl;
                wchar_t model[64] = { };
                itemIndex = SendMessage(g_hComboBox2, CB_GETCURSEL, 0, 0);
                textLength = SendMessage(g_hComboBox2, CB_GETLBTEXTLEN, itemIndex, 0);
                SendMessage(g_hComboBox2, CB_GETLBTEXT, itemIndex, (LPARAM)(LPCTSTR)model);

                std::wstring modelSimpan = model;

                g_hTextBox10.GetText(bufferText[10], bufferSize);

                g_hTextBox11.GetText(bufferText[11], bufferSize);

                g_hTextBox12.GetText(bufferText[12], bufferSize);


                std::wofstream outFile("model.txt");
                
                if (!outFile) {
                    std::cerr << "Failed to open the file." << std::endl;
                    return 1;
                }
                std::wcout << modelSimpan << std::endl;
                outFile << modelSimpan;
                outFile.close();



                LVITEMW lvItem;
                lvItem.mask = LVIF_TEXT;
                lvItem.iItem = ListView_GetItemCount(g_hListView); // Insert at the end
                lvItem.iSubItem = 0;
                lvItem.pszText = const_cast<LPWSTR>(L"Harga Rumah"); // Placeholder, you can set the actual value
                ListView_InsertItem(g_hListView, &lvItem);


                std::wstring join;
                join += L"['";
                for (int i = 0; i < 13; ++i) {
                    ListView_SetItemText(g_hListView, lvItem.iItem, i + 1, const_cast<LPWSTR>(bufferText[i]));
                    if (i > 0) {
                        join += L"', '";
                    }
                    join += bufferText[i];
                }
                join += L"']";



                const wchar_t* joinC = join.c_str();

                std::cout << joinC << std::endl;
                wprintf(joinC);
                wprintf(L"ls\n");

                int size_needed = WideCharToMultiByte(CP_UTF8, 0, join.c_str(), static_cast<int>(join.length()), nullptr, 0, nullptr, nullptr);
                std::string strTo(size_needed, 0);
                WideCharToMultiByte(CP_UTF8, 0, join.c_str(), static_cast<int>(join.length()), &strTo[0], size_needed, nullptr, nullptr);
                
                std::string convertedStr = "python predict.py \" " + strTo + "\"";


                int wh = std::system(convertedStr.c_str());
                std::cout << wh << std::endl;
                std::cout << convertedStr.c_str() << std::endl;

                wchar_t bufferREAD[64];
                std::wifstream file("hasil.txt");

                if (!file.is_open()) {
                    std::cerr << "Error opening file: " << "hasil.txt" << std::endl;
                    return 1;
                }

                file >> bufferREAD;

                file.close();


 


                std::wcsncpy(isiHasil, bufferREAD, 64-1);
                isiHasil[63] = L'\0';

                SetWindowTextW(hasil, isiHasil);

                ListView_SetItemText(g_hListView, lvItem.iItem, 0, const_cast<LPWSTR>(bufferREAD));

                MessageBoxW(hwnd, bufferREAD, L"Text Entered", MB_OK | MB_ICONINFORMATION);

                isiHasil[63] = L'\0';

                bool fileExists = std::filesystem::exists(L"output.csv");
                std::wofstream csvFile(L"output.csv", std::ios::app); // Open in append mode

                if (!csvFile.is_open()) {
                    std::wcerr << L"Could not open file for writing: " << L"output.csv" << std::endl;
                    return -1;
                }
                csvFile << L"\n"; // Add a newline at the end of the header row
                if (!fileExists) {
                    for (int i = 0; i < 14; ++i) {
                        csvFile << arr[i];
                        if (i < 13) {
                            csvFile << L","; // Add a comma between elements
                        }
                    }
                    csvFile << L"\n"; // Add a newline at the end of the header row
                }


                csvFile << bufferREAD;
                csvFile << L",";
                for (int i = 1; i < 14; ++i) {
                    std::wstring line(bufferText[i]);
                    csvFile << line;
                    if (i < 13 - 1) {
                        csvFile << L","; // Add a comma between elements
                    }
                }
                csvFile << std::endl; // Add a newline at the end of the row
                csvFile.close();

                //wchar_t* selectedText = new wchar_t[textLength + 1];
                //SendMessage(g_hComboBox, CB_GETLBTEXT, itemIndex, (LPARAM)(LPCTSTR)selectedText);
            }
            break;

        case WM_KEYDOWN:
        {
            POINT cursorPos;
            GetCursorPos(&cursorPos);
            ScreenToClient(hwnd, &cursorPos);
            std::cout << cursorPos.x << std::endl;
            std::cout << cursorPos.y << std::endl;
        }
            break;
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;

        case WM_PAINT:
        {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hwnd, &ps);
            FillRect(hdc, &ps.rcPaint, (HBRUSH)(COLOR_WINDOW + 1));
            EndPaint(hwnd, &ps);
        }
        default:
            return DefWindowProc(hwnd, uMsg, wParam, lParam);
        }
    return 0;
}

void RedirectIOToConsole()
{
    // Allocate a console for this application
    AllocConsole();

    // Redirect standard input, output, and error streams to the console
    FILE* fDummy;
    freopen_s(&fDummy, "CONIN$", "r", stdin);
    freopen_s(&fDummy, "CONOUT$", "w", stdout);
    freopen_s(&fDummy, "CONOUT$", "w", stderr);


    // Clear the error state for std::cout
    std::cout.clear();
}
